import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

import numpy as np

from .builder import MODELS, build_model
from .mask_matrix_nms import mask_matrix_nms

from torch_geometric.utils import scatter
from pointgroup_ops import voxelization
from ..custom_spconv_module.spconv_layers import SubMConv3d
from ..custom_spconv_module.batchnorm import BatchNorm1d
from ..spconv_unet import SpConvUNet
from pointops import farthest_point_sampling


@MODELS.register_module()
class OmniSuperPointTransformer(nn.Module):
    def __init__(self, 
                 backbone=None, 
                 decoder=None, 
                 query_thr=0.5, 
                 test_cfg=None, 
                 num_classes=200,
                 stuff_classes=[0, 1],
                 num_channels=32,
                 num_keep=200,
                 **kwargs):
        super().__init__()
    
        self._init_proj_layers()
        
        self.unet = SpConvUNet(
                num_planes=[num_channels * (i + 1) for i in range(5)],
                return_blocks=True)

        self.decoder = build_model(decoder)
        self.query_thr = query_thr
        self.test_cfg = test_cfg
        self.stuff_classes = stuff_classes

        self.thing_classes = np.array([i for i in range(num_classes) if i not in self.stuff_classes])
        
        self.num_inst_classes = len(self.thing_classes)
        
        self.num_keep = num_keep
        self.dvc_layout_cells_per_axis = kwargs.get("dvc_layout_cells_per_axis", 6)
        self.dvc_second_token_semantics = kwargs.get("dvc_second_token_semantics", "frame_anchor")
        self.dvc_num_boundary_anchors = kwargs.get("dvc_num_boundary_anchors", 4)
        self.dvc_region_grid_x = kwargs.get("dvc_region_grid_x", 3)
        self.dvc_region_grid_y = kwargs.get("dvc_region_grid_y", 3)
        self.dvc_rel_band_ratio = kwargs.get("dvc_rel_band_ratio", 0.15)
        self.dvc_rel_min_points = kwargs.get("dvc_rel_min_points", 4)
        self.dvc_variant = kwargs.get("dvc_variant", "legacy")
        self.tsp_use_ref_objects = kwargs.get("tsp_use_ref_objects", True)
        self.tsp_use_vertical_anchors = kwargs.get("tsp_use_vertical_anchors", True)
        self.tsp_ref_raw_tokens = kwargs.get("tsp_ref_raw_tokens", 8)
        self.tsp_struct_raw_tokens = kwargs.get("tsp_struct_raw_tokens", 8)
    
    def _init_proj_layers(self):        
        self.input_conv = spconv.SparseSequential(
            SubMConv3d(
                6,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))

        self.output_layer = spconv.SparseSequential(
            BatchNorm1d(32, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))
        
        self.pos_encode = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        self.alignment_proj = nn.Sequential(
            nn.Linear(256, 1024),
        )

    def extract_feat(self, input_dict, sp_pts_masks, sp_batch_offsets):
        batch_size = len(input_dict["offset"])
        grid_coord = input_dict["grid_coord"]
        feats = input_dict["feat"]
        v2p_map = input_dict["v2p_map"]
        spatial_shape = input_dict["spatial_shape"]
        xyz_all = input_dict["coord"]

        # to align with the pretrained backbone
        feats = torch.cat([feats[:, 3:], feats[:, :3] - feats[:, :3].mean(0)], dim=1)
        compute_dtype = feats.dtype
        
        with torch.cuda.amp.autocast(enabled=False):
            voxel_feats = voxelization(feats.float(), v2p_map)
            voxel_input = spconv.SparseConvTensor(
                features=voxel_feats,
                indices=grid_coord.int(),
                spatial_shape=spatial_shape,
                batch_size=batch_size,
            )
        
        voxel_input = voxel_input.replace_feature(voxel_input.features.to(compute_dtype).detach())
        
        with torch.cuda.amp.autocast(enabled=False):
            voxel_input = self.input_conv(voxel_input)
            voxel_feats, _ = self.unet(voxel_input)
            voxel_feats = self.output_layer(voxel_feats)
            voxel_feats = voxel_feats.features

        # get point features from voxel features
        p2v_map = input_dict["p2v_map"].long()
        x = voxel_feats[p2v_map]
        x_pos = self.pos_encode(xyz_all)
        # get superpoint features from point features
        with torch.cuda.amp.autocast(enabled=False):
            x = scatter(x.float(), sp_pts_masks.long(), reduce="mean", dim=0)
            x_pos = scatter(x_pos.float(), sp_pts_masks.long(), reduce="mean", dim=0)
            sp_xyz = scatter(xyz_all.float(), sp_pts_masks.long(), reduce="mean", dim=0)
            x = x + x_pos

        x = x.to(compute_dtype)
        sp_xyz = sp_xyz.to(compute_dtype)
        
        out = []
        out_xyz = []
        for i in range(len(sp_batch_offsets)-1):
            out.append(x[sp_batch_offsets[i]: sp_batch_offsets[i+1]])
            out_xyz.append(sp_xyz[sp_batch_offsets[i]: sp_batch_offsets[i+1]])
        
        return out, out_xyz
    
    def _select_queries(self, x, input_dict):
        """Select queries for train pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, n_channels).
            gt_sp_mask_list (List[Tensor]): of len batch_size.
                Groud truth of `sp_masks` of shape (n_gts_i, n_points_i).

        Returns:
            Tuple:
                List[Tensor]: Queries of len batch_size, each queries of shape
                    (n_queries_i, n_channels).
                List[InstanceData_]: of len batch_size, each updated
                    with `query_masks` of shape (n_gts_i, n_queries_i).
        """
        gt_sp_mask_list = input_dict["gt_inst_sp_masks"]
        queries = []
        gt_query_mask_list = []
        for i in range(len(x)):
            if self.query_thr < 1:
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).int()
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])
                gt_query_mask_list.append(gt_sp_mask_list[i][:, ids])
            else:
                queries.append(x[i])
                gt_query_mask_list.append(gt_sp_mask_list[i])
        return queries, gt_query_mask_list

    def predict_semantic(self, out_dict, sp_pts_masks, classes=None):
        if classes is None:
            classes = list(range(out_dict['sem_preds'][0].shape[1] - 1))
        return out_dict["sem_preds"][0][:, classes][sp_pts_masks]

    def predict_instance(self, out, sp_pts_masks, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        # score_threshold = self.test_cfg.inst_score_thr

        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        labels = torch.arange(
            self.num_inst_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_inst_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred_sigmoid = mask_pred_sigmoid[:, sp_pts_masks]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores


    def predict_panoptic(self, out, superpoints):
        """Predict panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        sem_logits = self.predict_semantic(
            out, superpoints, self.stuff_classes)
        sem_map = sem_logits.argmax(dim=1)

        mask_pred, labels, scores  = self.predict_instance(
            out, superpoints, self.test_cfg.pan_score_thr)
        if mask_pred.shape[0] == 0:
            return sem_map, sem_map
        
        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        n_stuff_classes = len(self.stuff_classes)
        inst_idxs = torch.arange(
            n_stuff_classes, 
            mask_pred.shape[0] + n_stuff_classes, 
            device=mask_pred.device).view(-1, 1)

        insts = inst_idxs * mask_pred

        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs] + n_stuff_classes

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map[things_inst_mask != 0] = 0
        inst_map = sem_map.clone()
        inst_map += things_inst_mask
        sem_map += things_sem_mask
        return sem_map, inst_map


    def _build_layout_candidates_from_cells(
            self,
            cur_sp_feat,
            cur_sp_xyz,
            exclude_idx,
            num_layout_tokens,
            cells_per_axis,
    ):
        empty_feat = cur_sp_feat.new_zeros((0, cur_sp_feat.shape[-1]))
        empty_xyz = cur_sp_xyz.new_zeros((0, cur_sp_xyz.shape[-1]))
        layout_meta = {
            "occupied_layout_cells": 0,
            "num_layout_candidates": 0,
            "avg_layout_cell_mass": 0.0,
            "max_layout_cell_mass": 0,
        }

        if num_layout_tokens <= 0 or cur_sp_feat.shape[0] == 0:
            return empty_feat, empty_xyz, layout_meta

        keep_mask = torch.ones(cur_sp_feat.shape[0], dtype=torch.bool, device=cur_sp_feat.device)
        if exclude_idx.numel() > 0:
            keep_mask[exclude_idx] = False

        layout_feat = cur_sp_feat[keep_mask]
        layout_xyz = cur_sp_xyz[keep_mask]
        if layout_feat.shape[0] == 0:
            return empty_feat, empty_xyz, layout_meta

        cells_per_axis = max(int(cells_per_axis), 1)
        xyz_min = layout_xyz.min(dim=0)[0]
        xyz_max = layout_xyz.max(dim=0)[0]
        xyz_extent = (xyz_max - xyz_min).clamp_min(1e-6)
        norm_xyz = (layout_xyz - xyz_min) / xyz_extent
        cell_coords = torch.floor(norm_xyz * cells_per_axis).long().clamp_(0, cells_per_axis - 1)

        unique_cells, inverse, cell_counts = torch.unique(
            cell_coords, dim=0, return_inverse=True, return_counts=True
        )

        layout_feat_candidates = scatter(layout_feat.float(), inverse, reduce="mean", dim=0)
        layout_xyz_candidates = scatter(layout_xyz.float(), inverse, reduce="mean", dim=0)
        layout_feat_candidates = layout_feat_candidates.to(cur_sp_feat.dtype)
        layout_xyz_candidates = layout_xyz_candidates.to(cur_sp_xyz.dtype)

        layout_meta = {
            "occupied_layout_cells": int(unique_cells.shape[0]),
            "num_layout_candidates": int(unique_cells.shape[0]),
            "avg_layout_cell_mass": float(cell_counts.float().mean().item()),
            "max_layout_cell_mass": int(cell_counts.max().item()),
        }
        return layout_feat_candidates, layout_xyz_candidates, layout_meta

    def _select_layout_tokens_by_fps(
            self,
            layout_feat_candidates,
            layout_xyz_candidates,
            num_layout_tokens,
    ):
        if num_layout_tokens <= 0 or layout_feat_candidates.shape[0] == 0:
            empty_feat = layout_feat_candidates.new_zeros((0, layout_feat_candidates.shape[-1]))
            empty_xyz = layout_xyz_candidates.new_zeros((0, layout_xyz_candidates.shape[-1]))
            empty_idx = torch.zeros(0, dtype=torch.long, device=layout_feat_candidates.device)
            return empty_feat, empty_xyz, empty_idx

        num_candidates = layout_feat_candidates.shape[0]
        num_keep = min(num_candidates, num_layout_tokens)
        if num_candidates <= num_layout_tokens:
            keep_idx = torch.arange(num_candidates, device=layout_feat_candidates.device)
            return layout_feat_candidates, layout_xyz_candidates, keep_idx

        coords = layout_xyz_candidates.float()
        selected_idx = torch.empty(num_keep, dtype=torch.long, device=coords.device)
        center = coords.mean(dim=0, keepdim=True)
        dist_to_center = ((coords - center) ** 2).sum(dim=1)
        selected_idx[0] = dist_to_center.argmax()

        min_dist = ((coords - coords[selected_idx[0]]) ** 2).sum(dim=1)
        for idx in range(1, num_keep):
            selected_idx[idx] = min_dist.argmax()
            current_dist = ((coords - coords[selected_idx[idx]]) ** 2).sum(dim=1)
            min_dist = torch.minimum(min_dist, current_dist)

        return (
            layout_feat_candidates[selected_idx],
            layout_xyz_candidates[selected_idx],
            selected_idx,
        )

    def _normalize_second_token_semantics(self, second_token_semantics, dvc_variant):
        semantics = str(second_token_semantics)
        if str(dvc_variant) not in {"tsp3d", "ssc3d"}:
            return semantics
        if semantics in {"generic_layout", "relation_anchor", "hybrid_ra", "gta_lite"}:
            return "frame_anchor"
        return semantics

    def _build_boundary_like_anchors(
            self,
            cur_sp_feat,
            cur_sp_xyz,
            exclude_idx,
            num_boundary_anchors,
            band_ratio,
            min_points,
    ):
        empty_feat = cur_sp_feat.new_zeros((0, cur_sp_feat.shape[-1]))
        empty_xyz = cur_sp_xyz.new_zeros((0, cur_sp_xyz.shape[-1]))
        empty_type = torch.zeros(0, dtype=torch.long, device=cur_sp_feat.device)
        boundary_meta = {
            "boundary_candidate_count": 0,
            "boundary_selected_count": 0,
            "empty_boundary_sides": ["left", "right", "front", "back"],
        }

        if num_boundary_anchors <= 0 or cur_sp_feat.shape[0] == 0:
            return empty_feat, empty_xyz, empty_type, boundary_meta

        keep_mask = torch.ones(cur_sp_xyz.shape[0], dtype=torch.bool, device=cur_sp_xyz.device)
        if exclude_idx.numel() > 0:
            keep_mask[exclude_idx] = False
        feat = cur_sp_feat[keep_mask]
        xyz = cur_sp_xyz[keep_mask]
        if feat.shape[0] == 0:
            return empty_feat, empty_xyz, empty_type, boundary_meta

        band_ratio = max(float(band_ratio), 0.0)
        min_points = max(int(min_points), 1)
        xyz_min = cur_sp_xyz.min(dim=0)[0]
        xyz_max = cur_sp_xyz.max(dim=0)[0]
        x_span = (xyz_max[0] - xyz_min[0]).clamp_min(1e-6)
        y_span = (xyz_max[1] - xyz_min[1]).clamp_min(1e-6)

        left_thr = xyz_min[0] + band_ratio * x_span
        right_thr = xyz_max[0] - band_ratio * x_span
        front_thr = xyz_min[1] + band_ratio * y_span
        back_thr = xyz_max[1] - band_ratio * y_span

        side_masks = [
            ("left", xyz[:, 0] <= left_thr, 0),
            ("right", xyz[:, 0] >= right_thr, 1),
            ("front", xyz[:, 1] <= front_thr, 2),
            ("back", xyz[:, 1] >= back_thr, 3),
        ]

        boundary_feat_list = []
        boundary_xyz_list = []
        boundary_type_list = []
        empty_boundary_sides = []

        for side_name, side_mask, side_type in side_masks:
            side_count = int(side_mask.sum().item())
            if side_count >= min_points:
                side_feat = feat[side_mask].float().mean(dim=0, keepdim=True).to(cur_sp_feat.dtype)
                side_xyz = xyz[side_mask].float().mean(dim=0, keepdim=True).to(cur_sp_xyz.dtype)
                boundary_feat_list.append(side_feat)
                boundary_xyz_list.append(side_xyz)
                boundary_type_list.append(
                    torch.tensor([side_type], dtype=torch.long, device=cur_sp_feat.device)
                )
            else:
                empty_boundary_sides.append(side_name)

        boundary_candidate_count = len(boundary_feat_list)
        if boundary_candidate_count == 0:
            boundary_meta["empty_boundary_sides"] = empty_boundary_sides
            return empty_feat, empty_xyz, empty_type, boundary_meta

        if boundary_candidate_count > int(num_boundary_anchors):
            keep = int(num_boundary_anchors)
            boundary_feat_list = boundary_feat_list[:keep]
            boundary_xyz_list = boundary_xyz_list[:keep]
            boundary_type_list = boundary_type_list[:keep]

        boundary_feat = torch.cat(boundary_feat_list, dim=0)
        boundary_xyz = torch.cat(boundary_xyz_list, dim=0)
        boundary_type = torch.cat(boundary_type_list, dim=0)

        boundary_meta = {
            "boundary_candidate_count": int(boundary_candidate_count),
            "boundary_selected_count": int(boundary_feat.shape[0]),
            "empty_boundary_sides": empty_boundary_sides,
        }
        return boundary_feat, boundary_xyz, boundary_type, boundary_meta

    def _build_region_like_anchors(
            self,
            cur_sp_feat,
            cur_sp_xyz,
            exclude_idx,
            grid_x,
            grid_y,
            num_region_keep,
            min_points,
    ):
        empty_feat = cur_sp_feat.new_zeros((0, cur_sp_feat.shape[-1]))
        empty_xyz = cur_sp_xyz.new_zeros((0, cur_sp_xyz.shape[-1]))
        empty_type = torch.zeros(0, dtype=torch.long, device=cur_sp_feat.device)
        region_meta = {
            "region_candidate_count": 0,
            "region_selected_count": 0,
            "occupied_region_cells": [],
        }

        if num_region_keep <= 0 or cur_sp_feat.shape[0] == 0:
            return empty_feat, empty_xyz, empty_type, region_meta

        keep_mask = torch.ones(cur_sp_xyz.shape[0], dtype=torch.bool, device=cur_sp_xyz.device)
        if exclude_idx.numel() > 0:
            keep_mask[exclude_idx] = False
        feat = cur_sp_feat[keep_mask]
        xyz = cur_sp_xyz[keep_mask]
        if feat.shape[0] == 0:
            return empty_feat, empty_xyz, empty_type, region_meta

        grid_x = max(int(grid_x), 1)
        grid_y = max(int(grid_y), 1)
        min_points = max(int(min_points), 1)

        xyz_min = cur_sp_xyz.min(dim=0)[0]
        xyz_max = cur_sp_xyz.max(dim=0)[0]
        x_span = (xyz_max[0] - xyz_min[0]).clamp_min(1e-6)
        y_span = (xyz_max[1] - xyz_min[1]).clamp_min(1e-6)

        x_norm = ((xyz[:, 0] - xyz_min[0]) / x_span).clamp(0.0, 1.0 - 1e-6)
        y_norm = ((xyz[:, 1] - xyz_min[1]) / y_span).clamp(0.0, 1.0 - 1e-6)
        cell_x = torch.floor(x_norm * grid_x).long().clamp_(0, grid_x - 1)
        cell_y = torch.floor(y_norm * grid_y).long().clamp_(0, grid_y - 1)
        cell_id = cell_y * grid_x + cell_x

        unique_cells, inverse, cell_counts = torch.unique(
            cell_id, return_inverse=True, return_counts=True
        )
        valid_cell_mask = cell_counts >= min_points
        if valid_cell_mask.sum().item() == 0:
            return empty_feat, empty_xyz, empty_type, region_meta

        cell_feat = scatter(feat.float(), inverse, reduce="mean", dim=0).to(cur_sp_feat.dtype)
        cell_xyz = scatter(xyz.float(), inverse, reduce="mean", dim=0).to(cur_sp_xyz.dtype)

        region_feat = cell_feat[valid_cell_mask]
        region_xyz = cell_xyz[valid_cell_mask]
        region_type = unique_cells[valid_cell_mask].long()
        region_counts = cell_counts[valid_cell_mask]

        region_candidate_count = int(region_feat.shape[0])
        occupied_region_cells = [int(cell) for cell in region_type.detach().cpu().tolist()]

        if region_candidate_count > int(num_region_keep):
            top_idx = torch.argsort(region_counts, descending=True)[:int(num_region_keep)]
            region_feat = region_feat[top_idx]
            region_xyz = region_xyz[top_idx]
            region_type = region_type[top_idx]

        region_meta = {
            "region_candidate_count": int(region_candidate_count),
            "region_selected_count": int(region_feat.shape[0]),
            "occupied_region_cells": occupied_region_cells,
        }
        return region_feat, region_xyz, region_type, region_meta

    def _build_relation_anchor_tokens(
            self,
            cur_sp_feat,
            cur_sp_xyz,
            exclude_idx,
            num_relation_tokens,
            num_boundary_anchors,
            grid_x,
            grid_y,
            band_ratio,
            min_points,
    ):
        empty_feat = cur_sp_feat.new_zeros((0, cur_sp_feat.shape[-1]))
        empty_xyz = cur_sp_xyz.new_zeros((0, cur_sp_xyz.shape[-1]))
        empty_type = torch.zeros(0, dtype=torch.long, device=cur_sp_feat.device)
        rel_meta = {
            "num_boundary_selected": 0,
            "num_region_selected": 0,
            "boundary_candidate_count": 0,
            "region_candidate_count": 0,
            "num_relation_tokens_valid": 0,
            "empty_boundary_sides": ["left", "right", "front", "back"],
            "occupied_region_cells": [],
            "selected_group_ids": [],
            "selected_group_kind": [],
        }

        num_relation_tokens = int(num_relation_tokens)
        if num_relation_tokens <= 0 or cur_sp_feat.shape[0] == 0:
            return empty_feat, empty_xyz, empty_type, rel_meta

        boundary_feat, boundary_xyz, boundary_type, boundary_meta = self._build_boundary_like_anchors(
            cur_sp_feat=cur_sp_feat,
            cur_sp_xyz=cur_sp_xyz,
            exclude_idx=exclude_idx,
            num_boundary_anchors=num_boundary_anchors,
            band_ratio=band_ratio,
            min_points=min_points,
        )

        num_boundary_selected = int(boundary_feat.shape[0])
        num_region_keep = max(num_relation_tokens - num_boundary_selected, 0)
        region_feat, region_xyz, region_type, region_meta = self._build_region_like_anchors(
            cur_sp_feat=cur_sp_feat,
            cur_sp_xyz=cur_sp_xyz,
            exclude_idx=exclude_idx,
            grid_x=grid_x,
            grid_y=grid_y,
            num_region_keep=num_region_keep,
            min_points=min_points,
        )

        rel_feat_parts = []
        rel_xyz_parts = []
        rel_type_parts = []
        rel_group_id_parts = []
        rel_group_kind_parts = []
        if boundary_feat.shape[0] > 0:
            rel_feat_parts.append(boundary_feat)
            rel_xyz_parts.append(boundary_xyz)
            rel_type_parts.append(boundary_type)
            rel_group_id_parts.append(boundary_type.long())
            rel_group_kind_parts.append(torch.zeros_like(boundary_type.long()))
        if region_feat.shape[0] > 0:
            rel_feat_parts.append(region_feat)
            rel_xyz_parts.append(region_xyz)
            rel_type_parts.append(region_type)
            rel_group_id_parts.append(region_type.long() + 4)
            rel_group_kind_parts.append(torch.ones_like(region_type.long()))

        if len(rel_feat_parts) == 0:
            rel_feat, rel_xyz, rel_type = empty_feat, empty_xyz, empty_type
            rel_group_ids = empty_type
            rel_group_kind = empty_type
        else:
            rel_feat = torch.cat(rel_feat_parts, dim=0)
            rel_xyz = torch.cat(rel_xyz_parts, dim=0)
            rel_type = torch.cat(rel_type_parts, dim=0)
            rel_group_ids = torch.cat(rel_group_id_parts, dim=0)
            rel_group_kind = torch.cat(rel_group_kind_parts, dim=0)

        if rel_feat.shape[0] > num_relation_tokens:
            keep_boundary = min(int(boundary_feat.shape[0]), num_relation_tokens)
            keep_region = max(num_relation_tokens - keep_boundary, 0)

            rel_feat_parts = []
            rel_xyz_parts = []
            rel_type_parts = []
            rel_group_id_parts = []
            rel_group_kind_parts = []
            if keep_boundary > 0:
                rel_feat_parts.append(boundary_feat[:keep_boundary])
                rel_xyz_parts.append(boundary_xyz[:keep_boundary])
                rel_type_parts.append(boundary_type[:keep_boundary])
                rel_group_id_parts.append(boundary_type[:keep_boundary].long())
                rel_group_kind_parts.append(torch.zeros_like(boundary_type[:keep_boundary].long()))
            if keep_region > 0:
                rel_feat_parts.append(region_feat[:keep_region])
                rel_xyz_parts.append(region_xyz[:keep_region])
                rel_type_parts.append(region_type[:keep_region])
                rel_group_id_parts.append(region_type[:keep_region].long() + 4)
                rel_group_kind_parts.append(torch.ones_like(region_type[:keep_region].long()))

            if len(rel_feat_parts) == 0:
                rel_feat, rel_xyz, rel_type = empty_feat, empty_xyz, empty_type
                rel_group_ids = empty_type
                rel_group_kind = empty_type
            else:
                rel_feat = torch.cat(rel_feat_parts, dim=0)
                rel_xyz = torch.cat(rel_xyz_parts, dim=0)
                rel_type = torch.cat(rel_type_parts, dim=0)
                rel_group_ids = torch.cat(rel_group_id_parts, dim=0)
                rel_group_kind = torch.cat(rel_group_kind_parts, dim=0)

        final_num_boundary_selected = min(int(boundary_feat.shape[0]), int(rel_feat.shape[0]))
        final_num_region_selected = max(int(rel_feat.shape[0]) - final_num_boundary_selected, 0)

        rel_meta = {
            "num_boundary_selected": int(final_num_boundary_selected),
            "num_region_selected": int(final_num_region_selected),
            "boundary_candidate_count": int(boundary_meta["boundary_candidate_count"]),
            "region_candidate_count": int(region_meta["region_candidate_count"]),
            "num_relation_tokens_valid": int(rel_feat.shape[0]),
            "empty_boundary_sides": boundary_meta["empty_boundary_sides"],
            "occupied_region_cells": region_meta["occupied_region_cells"],
            "selected_group_ids": [int(v) for v in rel_group_ids.detach().cpu().tolist()],
            "selected_group_kind": [int(v) for v in rel_group_kind.detach().cpu().tolist()],
        }
        return rel_feat, rel_xyz, rel_type, rel_meta

    def _build_reference_object_tokens(
            self,
            aligned_sp_feat,
            sp_xyz,
            objectness_scores,
            num_entity_keep,
            num_ref_tokens,
    ):
        num_ref_tokens = max(int(num_ref_tokens), 0)
        feat_dim = aligned_sp_feat.shape[-1]
        xyz_dim = sp_xyz.shape[-1]
        ref_feat = aligned_sp_feat.new_zeros((num_ref_tokens, feat_dim))
        ref_xyz = sp_xyz.new_zeros((num_ref_tokens, xyz_dim))
        ref_valid_mask = torch.zeros(num_ref_tokens, dtype=torch.bool, device=aligned_sp_feat.device)
        ref_meta = {
            "num_ref_tokens_target": int(num_ref_tokens),
            "num_ref_tokens_valid": 0,
        }

        if num_ref_tokens == 0 or aligned_sp_feat.shape[0] == 0:
            return ref_feat, ref_xyz, ref_valid_mask, ref_meta

        sorted_idx = torch.argsort(objectness_scores, descending=True)
        start_idx = max(int(num_entity_keep), 0)
        ref_candidate_idx = sorted_idx[start_idx:]
        if ref_candidate_idx.numel() == 0:
            return ref_feat, ref_xyz, ref_valid_mask, ref_meta

        num_valid = min(int(ref_candidate_idx.shape[0]), num_ref_tokens)
        chosen_idx = ref_candidate_idx[:num_valid]
        ref_feat[:num_valid] = aligned_sp_feat[chosen_idx]
        ref_xyz[:num_valid] = sp_xyz[chosen_idx]
        ref_valid_mask[:num_valid] = True
        ref_meta["num_ref_tokens_valid"] = int(num_valid)
        return ref_feat, ref_xyz, ref_valid_mask, ref_meta

    def _build_support_vertical_anchors(
            self,
            aligned_sp_feat,
            sp_xyz,
            exclude_idx,
            num_vertical_anchors=2,
            z_band_ratio=0.15,
            min_points=4,
    ):
        num_vertical_anchors = max(int(num_vertical_anchors), 0)
        feat_dim = aligned_sp_feat.shape[-1]
        xyz_dim = sp_xyz.shape[-1]
        vertical_feat = aligned_sp_feat.new_zeros((num_vertical_anchors, feat_dim))
        vertical_xyz = sp_xyz.new_zeros((num_vertical_anchors, xyz_dim))
        vertical_type = torch.full(
            (num_vertical_anchors,),
            -1,
            dtype=torch.long,
            device=aligned_sp_feat.device,
        )
        vertical_valid_mask = torch.zeros(
            num_vertical_anchors,
            dtype=torch.bool,
            device=aligned_sp_feat.device,
        )
        vertical_meta = {
            "num_vertical_selected": 0,
            "num_vertical_invalid": int(num_vertical_anchors),
            "vertical_candidate_count": 0,
        }

        if num_vertical_anchors == 0 or aligned_sp_feat.shape[0] == 0:
            return vertical_feat, vertical_xyz, vertical_type, vertical_valid_mask, vertical_meta

        keep_mask = torch.ones(sp_xyz.shape[0], dtype=torch.bool, device=sp_xyz.device)
        if exclude_idx.numel() > 0:
            keep_mask[exclude_idx] = False
        feat = aligned_sp_feat[keep_mask]
        xyz = sp_xyz[keep_mask]
        if feat.shape[0] == 0:
            return vertical_feat, vertical_xyz, vertical_type, vertical_valid_mask, vertical_meta

        min_points = max(int(min_points), 1)
        z_band_ratio = max(float(z_band_ratio), 0.0)
        z_min = sp_xyz[:, 2].min()
        z_max = sp_xyz[:, 2].max()
        z_span = (z_max - z_min).clamp_min(1e-6)
        lower_thr = z_min + z_band_ratio * z_span
        upper_thr = z_max - z_band_ratio * z_span

        anchor_masks = []
        for anchor_idx in range(num_vertical_anchors):
            if anchor_idx == 0:
                anchor_masks.append(xyz[:, 2] <= lower_thr)
            elif anchor_idx == 1:
                anchor_masks.append(xyz[:, 2] >= upper_thr)
            else:
                anchor_masks.append(torch.zeros_like(xyz[:, 2], dtype=torch.bool))

        selected = 0
        for anchor_idx, anchor_mask in enumerate(anchor_masks):
            vertical_type[anchor_idx] = 100 + anchor_idx
            if int(anchor_mask.sum().item()) >= min_points:
                vertical_feat[anchor_idx] = feat[anchor_mask].float().mean(dim=0).to(aligned_sp_feat.dtype)
                vertical_xyz[anchor_idx] = xyz[anchor_mask].float().mean(dim=0).to(sp_xyz.dtype)
                vertical_valid_mask[anchor_idx] = True
                selected += 1

        vertical_meta = {
            "num_vertical_selected": int(selected),
            "num_vertical_invalid": int(num_vertical_anchors - selected),
            "vertical_candidate_count": int(selected),
        }
        return vertical_feat, vertical_xyz, vertical_type, vertical_valid_mask, vertical_meta

    def _build_tsp_structural_anchors(
            self,
            aligned_sp_feat,
            sp_xyz,
            topk_head_idx,
            num_struct_tokens,
            num_boundary_anchors,
            grid_x,
            grid_y,
            band_ratio,
            min_points,
            use_vertical_anchors=True,
    ):
        num_struct_tokens = max(int(num_struct_tokens), 0)
        feat_dim = aligned_sp_feat.shape[-1]
        xyz_dim = sp_xyz.shape[-1]
        struct_feat = aligned_sp_feat.new_zeros((num_struct_tokens, feat_dim))
        struct_xyz = sp_xyz.new_zeros((num_struct_tokens, xyz_dim))
        struct_type = torch.full((num_struct_tokens,), -1, dtype=torch.long, device=aligned_sp_feat.device)
        struct_valid_mask = torch.zeros(num_struct_tokens, dtype=torch.bool, device=aligned_sp_feat.device)

        empty_meta = {
            "num_struct_tokens_target": int(num_struct_tokens),
            "num_struct_tokens_valid": 0,
            "num_boundary_selected": 0,
            "num_region_selected": 0,
            "num_vertical_selected": 0,
            "boundary_candidate_count": 0,
            "region_candidate_count": 0,
            "vertical_candidate_count": 0,
            "struct_padding_count": int(num_struct_tokens),
        }

        if num_struct_tokens == 0 or aligned_sp_feat.shape[0] == 0:
            return struct_feat, struct_xyz, struct_type, struct_valid_mask, empty_meta

        boundary_feat, boundary_xyz, boundary_type, boundary_meta = self._build_boundary_like_anchors(
            cur_sp_feat=aligned_sp_feat,
            cur_sp_xyz=sp_xyz,
            exclude_idx=topk_head_idx,
            num_boundary_anchors=num_boundary_anchors,
            band_ratio=band_ratio,
            min_points=min_points,
        )

        num_region_probe = max(int(grid_x) * int(grid_y), num_struct_tokens)
        region_feat, region_xyz, region_type, region_meta = self._build_region_like_anchors(
            cur_sp_feat=aligned_sp_feat,
            cur_sp_xyz=sp_xyz,
            exclude_idx=topk_head_idx,
            grid_x=grid_x,
            grid_y=grid_y,
            num_region_keep=num_region_probe,
            min_points=min_points,
        )

        if use_vertical_anchors:
            vertical_feat, vertical_xyz, vertical_type, vertical_valid_mask, vertical_meta = self._build_support_vertical_anchors(
                aligned_sp_feat=aligned_sp_feat,
                sp_xyz=sp_xyz,
                exclude_idx=topk_head_idx,
                num_vertical_anchors=2,
                z_band_ratio=band_ratio,
                min_points=min_points,
            )
        else:
            vertical_feat = aligned_sp_feat.new_zeros((0, feat_dim))
            vertical_xyz = sp_xyz.new_zeros((0, xyz_dim))
            vertical_type = torch.zeros(0, dtype=torch.long, device=aligned_sp_feat.device)
            vertical_valid_mask = torch.zeros(0, dtype=torch.bool, device=aligned_sp_feat.device)
            vertical_meta = {
                "num_vertical_selected": 0,
                "num_vertical_invalid": 0,
                "vertical_candidate_count": 0,
            }

        vertical_keep = vertical_valid_mask.nonzero(as_tuple=False).view(-1)
        vertical_valid_feat = vertical_feat[vertical_keep]
        vertical_valid_xyz = vertical_xyz[vertical_keep]
        vertical_valid_type = vertical_type[vertical_keep]

        reserve_vertical = 2 if bool(use_vertical_anchors) and num_struct_tokens >= 2 else 0
        reserve_boundary = min(int(num_boundary_anchors), max(num_struct_tokens - reserve_vertical, 0))
        reserve_region = max(num_struct_tokens - reserve_boundary - reserve_vertical, 0)

        available_boundary = int(boundary_feat.shape[0])
        available_region = int(region_feat.shape[0])
        available_vertical = int(vertical_valid_feat.shape[0])

        keep_boundary = min(available_boundary, reserve_boundary)
        keep_vertical = min(available_vertical, reserve_vertical)
        keep_region = min(available_region, reserve_region)

        leftover = max(num_struct_tokens - (keep_boundary + keep_vertical + keep_region), 0)
        if leftover > 0:
            extra_region = min(max(available_region - keep_region, 0), leftover)
            keep_region += extra_region
            leftover -= extra_region
        if leftover > 0:
            extra_boundary = min(max(available_boundary - keep_boundary, 0), leftover)
            keep_boundary += extra_boundary
            leftover -= extra_boundary
        if leftover > 0:
            extra_vertical = min(max(available_vertical - keep_vertical, 0), leftover)
            keep_vertical += extra_vertical
            leftover -= extra_vertical

        write_cursor = 0
        if keep_boundary > 0:
            struct_feat[write_cursor:write_cursor + keep_boundary] = boundary_feat[:keep_boundary]
            struct_xyz[write_cursor:write_cursor + keep_boundary] = boundary_xyz[:keep_boundary]
            struct_type[write_cursor:write_cursor + keep_boundary] = boundary_type[:keep_boundary]
            struct_valid_mask[write_cursor:write_cursor + keep_boundary] = True
            write_cursor += keep_boundary
        if keep_region > 0:
            struct_feat[write_cursor:write_cursor + keep_region] = region_feat[:keep_region]
            struct_xyz[write_cursor:write_cursor + keep_region] = region_xyz[:keep_region]
            struct_type[write_cursor:write_cursor + keep_region] = region_type[:keep_region]
            struct_valid_mask[write_cursor:write_cursor + keep_region] = True
            write_cursor += keep_region
        if keep_vertical > 0:
            struct_feat[write_cursor:write_cursor + keep_vertical] = vertical_valid_feat[:keep_vertical]
            struct_xyz[write_cursor:write_cursor + keep_vertical] = vertical_valid_xyz[:keep_vertical]
            struct_type[write_cursor:write_cursor + keep_vertical] = vertical_valid_type[:keep_vertical]
            struct_valid_mask[write_cursor:write_cursor + keep_vertical] = True
            write_cursor += keep_vertical

        num_valid = int(struct_valid_mask.sum().item())
        struct_meta = {
            "num_struct_tokens_target": int(num_struct_tokens),
            "num_struct_tokens_valid": int(num_valid),
            "num_boundary_selected": int(keep_boundary),
            "num_region_selected": int(keep_region),
            "num_vertical_selected": int(keep_vertical),
            "boundary_candidate_count": int(boundary_meta["boundary_candidate_count"]),
            "region_candidate_count": int(region_meta["region_candidate_count"]),
            "vertical_candidate_count": int(vertical_meta["vertical_candidate_count"]),
            "struct_padding_count": int(num_struct_tokens - num_valid),
        }
        return struct_feat, struct_xyz, struct_type, struct_valid_mask, struct_meta

    def forward(
            self,
            input_dict,
            num_entity_tokens=100,
            num_layout_tokens=0,
            dvc_enable=False,
            dvc_second_token_semantics="frame_anchor",
            dvc_num_boundary_anchors=4,
            dvc_region_grid_x=3,
            dvc_region_grid_y=3,
            dvc_rel_band_ratio=0.15,
            dvc_rel_min_points=4,
            dvc_variant="legacy",
            tsp_use_ref_objects=True,
            tsp_use_vertical_anchors=True,
            tsp_ref_raw_tokens=8,
            tsp_struct_raw_tokens=8,
    ):
        batch_size = len(input_dict["offset"])
        sp_pts_mask_all = input_dict["superpoint_mask"]

        superpoint_bias = 0
        sp_pts_mask_list = []
        sp_batch_offsets = [0]
        for i in range(batch_size):
            sp_pts_mask = input_dict["superpoint_mask"][i]
            sp_pts_mask += superpoint_bias
            superpoint_bias += len(sp_pts_mask.unique())
            # superpoint_bias = sp_pts_mask.max().item() + 1
            sp_batch_offsets.append(superpoint_bias)
            sp_pts_mask_list.append(sp_pts_mask)
            
        sp_pts_masks = torch.hstack(sp_pts_mask_list)
        
        x, sp_xyz = self.extract_feat(input_dict, sp_pts_masks, sp_batch_offsets)

        queries = x
        out_dict = self.decoder(x, queries, sp_xyz)

        hidden_states = out_dict.pop("hidden_states")
        last_hidden_state = hidden_states[-1]
        aligned_sp_feat = [self.alignment_proj(cur_x) for cur_x in last_hidden_state]

        entity_query_feat_list = []
        entity_query_coord_list = []
        frame_anchor_feat_list = []
        frame_anchor_coord_list = []
        dvc_meta_list = []

        # Keep top-K objectness tokens as the explicit entity evidence bank.
        for i in range(batch_size):
            with torch.no_grad():
                cls_preds = out_dict["cls_preds"][i]
                scores = F.softmax(cls_preds, dim=-1)[:, :-1]
                max_scores = scores.max(1)[0]
                num_keep = min(max_scores.shape[0], int(num_entity_tokens))
                _, topk_idx = max_scores.topk(num_keep, sorted=True)

            entity_query_feat = aligned_sp_feat[i][topk_idx]
            entity_query_coord = sp_xyz[i][topk_idx]
            entity_query_feat_list.append(entity_query_feat)
            entity_query_coord_list.append(entity_query_coord)

            frame_anchor_feat = aligned_sp_feat[i].new_zeros((0, aligned_sp_feat[i].shape[-1]))
            frame_anchor_coord = sp_xyz[i].new_zeros((0, sp_xyz[i].shape[-1]))
            relation_meta = {
                "num_boundary_selected": 0,
                "num_region_selected": 0,
                "boundary_candidate_count": 0,
                "region_candidate_count": 0,
                "num_relation_tokens_valid": 0,
                "empty_boundary_sides": ["left", "right", "front", "back"],
                "occupied_region_cells": [],
                "selected_group_ids": [],
                "selected_group_kind": [],
            }

            scene_variant = str(dvc_variant)
            if scene_variant in {"tsp3d", "ssc3d"}:
                second_token_semantics = "frame_anchor"
            else:
                second_token_semantics = self._normalize_second_token_semantics(
                    dvc_second_token_semantics,
                    scene_variant,
                )
            num_frame_slots = int(num_layout_tokens if dvc_enable else 0)
            num_entity_keep = max(int(num_entity_tokens) - num_frame_slots, 0)
            num_entity_keep = min(num_entity_keep, int(entity_query_feat.shape[0]))

            if bool(dvc_enable) and num_frame_slots > 0:
                if scene_variant == "tsp3d":
                    frame_anchor_feat, frame_anchor_coord, _, relation_meta = self._build_relation_anchor_tokens(
                        cur_sp_feat=aligned_sp_feat[i],
                        cur_sp_xyz=sp_xyz[i],
                        exclude_idx=topk_idx,
                        num_relation_tokens=num_frame_slots,
                        num_boundary_anchors=dvc_num_boundary_anchors,
                        grid_x=dvc_region_grid_x,
                        grid_y=dvc_region_grid_y,
                        band_ratio=dvc_rel_band_ratio,
                        min_points=dvc_rel_min_points,
                    )
                elif scene_variant == "ssc3d":
                    empty_exclude = topk_idx.new_zeros((0,), dtype=torch.long)
                    frame_anchor_feat, frame_anchor_coord, _, relation_meta = self._build_relation_anchor_tokens(
                        cur_sp_feat=aligned_sp_feat[i],
                        cur_sp_xyz=sp_xyz[i],
                        exclude_idx=empty_exclude,
                        num_relation_tokens=num_frame_slots,
                        num_boundary_anchors=dvc_num_boundary_anchors,
                        grid_x=dvc_region_grid_x,
                        grid_y=dvc_region_grid_y,
                        band_ratio=dvc_rel_band_ratio,
                        min_points=dvc_rel_min_points,
                    )
                else:
                    use_frame_anchors = second_token_semantics in {"relation_anchor", "hybrid_ra", "gta_lite", "frame_anchor"}
                    if use_frame_anchors:
                        frame_anchor_feat, frame_anchor_coord, _, relation_meta = self._build_relation_anchor_tokens(
                            cur_sp_feat=aligned_sp_feat[i],
                            cur_sp_xyz=sp_xyz[i],
                            exclude_idx=topk_idx,
                            num_relation_tokens=num_frame_slots,
                            num_boundary_anchors=dvc_num_boundary_anchors,
                            grid_x=dvc_region_grid_x,
                            grid_y=dvc_region_grid_y,
                            band_ratio=dvc_rel_band_ratio,
                            min_points=dvc_rel_min_points,
                        )
                    else:
                        layout_feat_candidates, layout_xyz_candidates, _ = self._build_layout_candidates_from_cells(
                            aligned_sp_feat[i],
                            sp_xyz[i],
                            topk_idx,
                            num_frame_slots,
                            self.dvc_layout_cells_per_axis,
                        )
                        frame_anchor_feat, frame_anchor_coord, _ = self._select_layout_tokens_by_fps(
                            layout_feat_candidates,
                            layout_xyz_candidates,
                            num_frame_slots,
                        )

            relation_meta["num_relation_tokens_valid"] = int(frame_anchor_feat.shape[0])
            if (
                bool(dvc_enable)
                and scene_variant not in {"tsp3d", "ssc3d"}
                and num_frame_slots > 0
                and int(frame_anchor_feat.shape[0]) < num_frame_slots
            ):
                pad_count = int(num_frame_slots - frame_anchor_feat.shape[0])
                pad_feat = frame_anchor_feat.new_zeros((pad_count, frame_anchor_feat.shape[-1]))
                pad_coord = frame_anchor_coord.new_zeros((pad_count, frame_anchor_coord.shape[-1]))
                frame_anchor_feat = torch.cat([frame_anchor_feat, pad_feat], dim=0)
                frame_anchor_coord = torch.cat([frame_anchor_coord, pad_coord], dim=0)
                relation_meta["selected_group_ids"] = list(relation_meta.get("selected_group_ids", [])) + [-1] * pad_count
                relation_meta["selected_group_kind"] = list(relation_meta.get("selected_group_kind", [])) + [-1] * pad_count

            scene_second_semantics = "frame_anchor" if scene_variant in {"tsp3d", "ssc3d"} else str(second_token_semantics)
            scene_num_frame_valid = (
                int(relation_meta.get("num_relation_tokens_valid", frame_anchor_feat.shape[0]))
                if scene_variant in {"tsp3d", "ssc3d"}
                else int(relation_meta.get("num_relation_tokens_valid", frame_anchor_feat.shape[0]))
            )
            selected_group_ids = [int(v) for v in relation_meta.get("selected_group_ids", [])]
            selected_group_kind = [int(v) for v in relation_meta.get("selected_group_kind", [])]
            target_group_len = int(frame_anchor_feat.shape[0])
            if len(selected_group_ids) < target_group_len:
                selected_group_ids = selected_group_ids + [-1] * (target_group_len - len(selected_group_ids))
            if len(selected_group_kind) < target_group_len:
                selected_group_kind = selected_group_kind + [-1] * (target_group_len - len(selected_group_kind))
            selected_group_ids = selected_group_ids[:target_group_len]
            selected_group_kind = selected_group_kind[:target_group_len]
            frame_anchor_feat_list.append(frame_anchor_feat)
            frame_anchor_coord_list.append(frame_anchor_coord)
            dvc_meta_list.append(
                {
                    "total_tokens": int(entity_query_feat.shape[0]),
                    "num_entity_keep": int(num_entity_keep),
                    "num_frame_slots": int(num_frame_slots),
                    "ssc_frame_primitive_budget": int(num_frame_slots) if scene_variant == "ssc3d" else 0,
                    "num_frame_valid": int(scene_num_frame_valid),
                    "second_token_semantics": str(scene_second_semantics),
                    "selected_group_ids": selected_group_ids,
                    "selected_group_kind": selected_group_kind,
                    "num_boundary_selected": int(relation_meta["num_boundary_selected"]),
                    "num_region_selected": int(relation_meta["num_region_selected"]),
                    "boundary_candidate_count": int(relation_meta["boundary_candidate_count"]),
                    "region_candidate_count": int(relation_meta["region_candidate_count"]),
                }
            )

        return (
            entity_query_feat_list,
            entity_query_coord_list,
            frame_anchor_feat_list,
            frame_anchor_coord_list,
            aligned_sp_feat,
            x,
            sp_xyz,
            hidden_states[:-1],
            dvc_meta_list,
        )

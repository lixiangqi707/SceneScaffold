import torch
import torch.nn as nn
import spconv.pytorch as spconv
from torch_geometric.utils import scatter
from .custom_spconv_module.spconv_layers import SubMConv3d
from .custom_spconv_module.batchnorm import BatchNorm1d
import spconv.pytorch as spconv
import pathlib
from llava.utils import load_config
from .ost_module import build_model, build_criteria
from ..ssc_utils import normalize_ssc_ablation_mode, resolve_ssc_ablation_layout
import copy


class SPConvPointCloudTower(nn.Module):
    def __init__(self, pointcloud_tower, args, hidden_dim=32, delay_load=False):
        super().__init__()

        self.is_loaded = False
        
        self.pointcloud_tower_name = pointcloud_tower
        self.num_pc_tokens = args.num_pc_tokens
        self.dvc_enable = getattr(args, 'dvc_enable', False)
        self.dvc_entity_ratio = getattr(args, 'dvc_entity_ratio', 0.7)
        self.dvc_layout_cells_per_axis = getattr(args, 'dvc_layout_cells_per_axis', 6)
        self.dvc_role_embed = getattr(args, 'dvc_role_embed', True)
        self.dvc_second_token_semantics = getattr(args, 'dvc_second_token_semantics', 'frame_anchor')
        self.dvc_num_boundary_anchors = getattr(args, 'dvc_num_boundary_anchors', 4)
        self.dvc_region_grid_x = getattr(args, 'dvc_region_grid_x', 3)
        self.dvc_region_grid_y = getattr(args, 'dvc_region_grid_y', 3)
        self.dvc_rel_band_ratio = getattr(args, 'dvc_rel_band_ratio', 0.15)
        self.dvc_rel_min_points = getattr(args, 'dvc_rel_min_points', 4)
        self.dvc_variant = getattr(args, 'dvc_variant', 'legacy')
        self.ssc_state_ent_slots = getattr(args, 'ssc_state_ent_slots', 8)
        self.ssc_state_frame_slots = getattr(args, 'ssc_state_frame_slots', 8)
        self.ssc_state_rel_slots = getattr(args, 'ssc_state_rel_slots', 8)
        self.ssc_scene_summary_token_count = getattr(args, 'ssc_scene_summary_token_count', 1)
        self.ssc_detail_tokens = getattr(args, 'ssc_detail_tokens', -1)
        self.ssc_ablation_mode = normalize_ssc_ablation_mode(getattr(args, 'ssc_ablation_mode', 'none'))
        self.tsp_use_ref_objects = getattr(args, 'tsp_use_ref_objects', True)
        self.tsp_use_vertical_anchors = getattr(args, 'tsp_use_vertical_anchors', True)
        self.tsp_ref_raw_tokens = getattr(args, 'tsp_ref_raw_tokens', 8)
        self.tsp_struct_raw_tokens = getattr(args, 'tsp_struct_raw_tokens', 8)
        if "llava_distill" in self.pointcloud_tower_name or \
            "align" in self.pointcloud_tower_name:
            self.hidden_dim = 1024
        else:
            self.hidden_dim = hidden_dim

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_pointcloud_tower', False):
            self.load_model()
        else:
            self.cfg_only = None

    def _normalize_second_token_semantics(self, semantics):
        semantics = str(semantics)
        if str(self.dvc_variant) not in {"tsp3d", "ssc3d"}:
            return semantics
        if semantics in {"generic_layout", "relation_anchor", "hybrid_ra", "gta_lite"}:
            return "frame_anchor"
        return semantics

    def _resolve_ssc_layout(self):
        return resolve_ssc_ablation_layout(
            num_pc_tokens=self.num_pc_tokens,
            ssc_state_ent_slots=self.ssc_state_ent_slots,
            ssc_state_frame_slots=self.ssc_state_frame_slots,
            ssc_state_rel_slots=self.ssc_state_rel_slots,
            ssc_scene_summary_token_count=self.ssc_scene_summary_token_count,
            ssc_detail_tokens=self.ssc_detail_tokens,
            ssc_ablation_mode=self.ssc_ablation_mode,
        )

    def _resolve_ssc_detail_tokens(self):
        return int(self._resolve_ssc_layout()["effective_detail_tokens"])

    def load_model(self, device_map=None, pointcloud_tower_name=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.video_tower_name))
            return
        
        if pointcloud_tower_name is not None:
            self.pointcloud_tower_name = pointcloud_tower_name

        config_name = self.pointcloud_tower_name.split('/')[-1]
        config_name = config_name.split('.')[0]

        current_file_path = pathlib.Path(__file__).resolve()
        current_dir = current_file_path.parent
        config_path = current_dir / 'configs' / f'{config_name}.py'
        cfg = load_config(config_path)

        self.segmentor = build_model(cfg.model)
        self.segmentor.dvc_layout_cells_per_axis = self.dvc_layout_cells_per_axis
        self.segmentor.dvc_second_token_semantics = self.dvc_second_token_semantics
        self.segmentor.dvc_num_boundary_anchors = self.dvc_num_boundary_anchors
        self.segmentor.dvc_region_grid_x = self.dvc_region_grid_x
        self.segmentor.dvc_region_grid_y = self.dvc_region_grid_y
        self.segmentor.dvc_rel_band_ratio = self.dvc_rel_band_ratio
        self.segmentor.dvc_rel_min_points = self.dvc_rel_min_points
        self.segmentor.dvc_variant = self.dvc_variant
        self.segmentor.tsp_use_ref_objects = self.tsp_use_ref_objects
        self.segmentor.tsp_use_vertical_anchors = self.tsp_use_vertical_anchors
        self.segmentor.tsp_ref_raw_tokens = self.tsp_ref_raw_tokens
        self.segmentor.tsp_struct_raw_tokens = self.tsp_struct_raw_tokens

        self.load_segmentor_weights()
        
        self.alignment_proj = self.segmentor.alignment_proj

        # re-use OST as the visual sampler
        self.visual_sampler = build_model(cfg.visual_sampler)

        self.visual_sampler.load_state_dict(
            copy.deepcopy(self.segmentor.decoder.state_dict()), 
            strict=False
            )

        for p in self.visual_sampler.parameters():
            p.requires_grad = False

        # re-use OST as the mask decoder
        self.mask_decoder = build_model(cfg.mask_decoder)

        self.mask_decoder.load_state_dict(
            copy.deepcopy(self.segmentor.decoder.state_dict()), 
            strict=False
            )

        for p in self.mask_decoder.parameters():
            p.requires_grad = True

        # projection layer for [SEG] tokens
        seg_fc = [
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.Dropout(0.0),
        ]
        self.hidden_seg_fc = nn.Sequential(*seg_fc)
        self.hidden_seg_fc.train()
        for param in self.hidden_seg_fc.parameters():
            param.requires_grad = True

        # segmentation criterion
        self.seg_criteria = build_criteria(cfg.criteria)

        self.is_loaded = True
    
    def load_segmentor_weights(self):
        ckpt_dir = self.pointcloud_tower_name.replace("_fp32", "")
        checkpoint = torch.load(ckpt_dir, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        checkpoint_to_load = dict()

        for name, val in checkpoint.items():
            checkpoint_to_load[name.replace("module.", "")] = val
        missing, unexpected = self.segmentor.load_state_dict(checkpoint_to_load, strict=False)

    def forward(self, coord, grid_coord, offset, feat, p2v_map, v2p_map, spatial_shape, superpoint_mask, prompt_mask):
        segmentor_input_dict = dict(
            coord=coord,
            grid_coord=grid_coord,
            offset=offset,
            feat=feat,
            p2v_map=p2v_map,
            v2p_map=v2p_map,
            spatial_shape=spatial_shape,
            superpoint_mask=superpoint_mask,
        )

        normalized_second_semantics = self._normalize_second_token_semantics(self.dvc_second_token_semantics)

        if self.dvc_enable and str(self.dvc_variant) == "ssc3d":
            ssc_layout = self._resolve_ssc_layout()
            resolved_detail_tokens = int(ssc_layout["effective_detail_tokens"])
            num_entity_keep = resolved_detail_tokens
            num_entity_tokens = resolved_detail_tokens if resolved_detail_tokens > 0 else 1
            num_frame_primitive_budget = (
                max(
                    int(ssc_layout["effective_frame_slots"]),
                    int(self.dvc_num_boundary_anchors) + int(self.dvc_region_grid_x) * int(self.dvc_region_grid_y),
                )
                if bool(ssc_layout["use_frame"])
                else 0
            )
            num_frame_slots = num_frame_primitive_budget
            num_layout_tokens = num_frame_primitive_budget
        elif self.dvc_enable:
            resolved_detail_tokens = 0
            num_entity_keep = int(round(self.num_pc_tokens * self.dvc_entity_ratio))
            num_entity_keep = max(1, min(num_entity_keep, self.num_pc_tokens))
            num_frame_slots = self.num_pc_tokens - num_entity_keep
            num_entity_tokens = self.num_pc_tokens
            num_layout_tokens = num_frame_slots
        else:
            resolved_detail_tokens = 0
            num_entity_keep = self.num_pc_tokens
            num_frame_slots = 0
            num_entity_tokens = self.num_pc_tokens
            num_layout_tokens = 0

        baseline_entity_features_full, entity_query_coord, frame_anchor_feat, frame_anchor_coord, aligned_sp_feat, sp_feature, sp_xyz, hidden_states, dvc_meta = self.segmentor(
            segmentor_input_dict,
            num_entity_tokens=num_entity_tokens,
            num_layout_tokens=num_layout_tokens,
            dvc_enable=self.dvc_enable,
            dvc_second_token_semantics=normalized_second_semantics,
            dvc_num_boundary_anchors=self.dvc_num_boundary_anchors,
            dvc_region_grid_x=self.dvc_region_grid_x,
            dvc_region_grid_y=self.dvc_region_grid_y,
            dvc_rel_band_ratio=self.dvc_rel_band_ratio,
            dvc_rel_min_points=self.dvc_rel_min_points,
            dvc_variant=self.dvc_variant,
            tsp_use_ref_objects=self.tsp_use_ref_objects,
            tsp_use_vertical_anchors=self.tsp_use_vertical_anchors,
            tsp_ref_raw_tokens=self.tsp_ref_raw_tokens,
            tsp_struct_raw_tokens=self.tsp_struct_raw_tokens,
        )
        
        prompt_feature = self.visual_sampler(prompt_mask, hidden_states, sp_xyz)
        prompt_feature = [self.alignment_proj(ff) for ff in prompt_feature]

        # SSC3D source-of-truth fields:
        # - ssc_detail_tokens_resolved: serialized detail buffer budget.
        # - ssc_baseline_entity_request_tokens: actual baseline entity path request.
        # - ssc_frame_primitive_budget: frame primitive parser budget.
        # In SSC3D, num_entity_keep/num_frame_slots are compatibility fields, not logic source-of-truth.
        dvc_meta = [
            {
                "total_tokens": int(scene_meta.get("total_tokens", self.num_pc_tokens)),
                "num_entity_keep": int(num_entity_keep if str(self.dvc_variant) == "ssc3d" else scene_meta.get("num_entity_keep", num_entity_keep)),
                "num_frame_slots": int(scene_meta.get("num_frame_slots", num_frame_slots)),
                "ssc_frame_primitive_budget": int(
                    scene_meta.get("ssc_frame_primitive_budget", scene_meta.get("num_frame_slots", 0))
                ),
                "ssc_detail_tokens_resolved": int(resolved_detail_tokens if str(self.dvc_variant) == "ssc3d" else 0),
                "ssc_baseline_entity_request_tokens": int(num_entity_tokens if str(self.dvc_variant) == "ssc3d" else 0),
                "ssc_ablation_mode": str(self.ssc_ablation_mode if str(self.dvc_variant) == "ssc3d" else "none"),
                "ssc_effective_frame_slots": int(
                    ssc_layout["effective_frame_slots"] if str(self.dvc_variant) == "ssc3d" else 0
                ),
                "ssc_effective_ent_slots": int(
                    ssc_layout["effective_ent_slots"] if str(self.dvc_variant) == "ssc3d" else 0
                ),
                "ssc_effective_rel_slots": int(
                    ssc_layout["effective_rel_slots"] if str(self.dvc_variant) == "ssc3d" else 0
                ),
                "ssc_scene_summary_token_count": int(
                    ssc_layout["effective_scene_summary_token_count"] if str(self.dvc_variant) == "ssc3d" else 0
                ),
                "num_frame_valid": int(scene_meta.get("num_frame_valid", 0)),
                "second_token_semantics": str(scene_meta.get("second_token_semantics", normalized_second_semantics)),
                "selected_group_ids": [int(v) for v in scene_meta.get("selected_group_ids", [])],
                "selected_group_kind": [int(v) for v in scene_meta.get("selected_group_kind", [])],
                "num_boundary_selected": int(scene_meta.get("num_boundary_selected", 0)),
                "num_region_selected": int(scene_meta.get("num_region_selected", 0)),
                "boundary_candidate_count": int(scene_meta.get("boundary_candidate_count", 0)),
                "region_candidate_count": int(scene_meta.get("region_candidate_count", 0)),
                **({"ssc_legacy_compat_only": True} if str(self.dvc_variant) == "ssc3d" else {}),
            }
            for scene_meta, cur_frame in zip(dvc_meta, frame_anchor_feat)
        ]
        frame_valid_counts = [int(scene_meta.get("num_frame_valid", 0)) for scene_meta in dvc_meta]
        ssc_frame_primitive_budget = [int(scene_meta.get("ssc_frame_primitive_budget", 0)) for scene_meta in dvc_meta]
        ssc_detail_tokens_resolved = [int(scene_meta.get("ssc_detail_tokens_resolved", 0)) for scene_meta in dvc_meta]
        ssc_baseline_entity_request_tokens = [int(scene_meta.get("ssc_baseline_entity_request_tokens", 0)) for scene_meta in dvc_meta]
        
        # SSC3D must use the explicit source-of-truth fields below:
        # - ssc_detail_tokens_resolved for serialized detail buffer budget.
        # - ssc_baseline_entity_request_tokens for baseline entity path request size.
        # - ssc_frame_primitive_budget for frame primitive parser budget.
        # - frame_valid_counts for real returned frame primitive evidence count.
        # Legacy fields total_tokens/num_entity_keep/num_frame_slots are compatibility-only
        # in SSC3D and must not drive state construction or analysis.
        mask_input_dict = {
            "sp_features": sp_feature,
            "sp_xyz": sp_xyz,
            "hidden_states": hidden_states,
            "dvc_meta": dvc_meta,
            "frame_anchor_coord": frame_anchor_coord,
            "second_token_semantics": [str(scene_meta.get("second_token_semantics", normalized_second_semantics)) for scene_meta in dvc_meta],
            "entity_token_counts": [int(feat.shape[0]) for feat in baseline_entity_features_full],
            # Returned frame anchor tensor length; use frame_valid_counts for the real valid count.
            "frame_token_counts": [int(feat.shape[0]) for feat in frame_anchor_feat],
            "frame_valid_counts": frame_valid_counts,
            "ssc_frame_primitive_budget": ssc_frame_primitive_budget,
            "ssc_detail_tokens_resolved": ssc_detail_tokens_resolved,
            "ssc_baseline_entity_request_tokens": ssc_baseline_entity_request_tokens,
            "ssc_ablation_mode": str(self.ssc_ablation_mode if str(self.dvc_variant) == "ssc3d" else "none"),
            "ssc_source_of_truth_fields": {
                "detail_tokens": "ssc_detail_tokens_resolved",
                "baseline_entity_request_tokens": "ssc_baseline_entity_request_tokens",
                "frame_primitive_budget": "ssc_frame_primitive_budget",
                "frame_valid_counts": "frame_valid_counts",
            },
            "frame_group_ids": [scene_meta.get("selected_group_ids", []) for scene_meta in dvc_meta],
            "frame_group_kind": [scene_meta.get("selected_group_kind", []) for scene_meta in dvc_meta],
            "boundary_selected_counts": [int(scene_meta.get("num_boundary_selected", 0)) for scene_meta in dvc_meta],
            "region_selected_counts": [int(scene_meta.get("num_region_selected", 0)) for scene_meta in dvc_meta],
        }
        return baseline_entity_features_full, frame_anchor_feat, prompt_feature, aligned_sp_feat, mask_input_dict
        
    @property
    def seg_query_dim(self):
        # hidden_seg_fc = [Linear(4096,1024), ReLU, Linear(1024,256), Dropout]
        # Infer the final query width from the last Linear layer.
        last_linear = None
        for module in self.hidden_seg_fc.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            raise RuntimeError("hidden_seg_fc has no Linear layer to infer seg_query_dim.")
        return int(last_linear.out_features)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.unet.dtype

    @property
    def device(self):
        return self.unet.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.hidden_dim

    @property
    def feature_dim(self):
        return self.hidden_dim

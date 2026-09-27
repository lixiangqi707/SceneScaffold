#    Modified from LLaVA repository
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import (build_vision_tower, 
                                         build_pointcloud_tower, 
                                         build_inst_prompt_encoder)
from .multimodal_projector.builder import build_vision_projector
from .ssc_utils import (
    normalize_ssc_ablation_mode,
    normalize_ssc_role_analysis_variant,
    resolve_ssc_ablation_layout,
    resolve_ssc_role_analysis_settings,
)

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, LOC_TOKEN_INDEX

from llava.mm_utils import get_anyres_image_grid_shape


class GatedResidualAdapter(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, gate_scale=1.0):
        return x + (gate_scale * self.gate) * self.fc2(self.act(self.fc1(self.norm(x))))


class SlotWiseMixGate(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size * 4)
        self.fc1 = nn.Linear(hidden_size * 4, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, entity_tail, layout_target):
        x = torch.cat(
            [
                entity_tail,
                layout_target,
                layout_target - entity_tail,
                layout_target * entity_tail,
            ],
            dim=-1,
        )
        return self.fc2(self.act(self.fc1(self.norm(x))))


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class TSPStateInductionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, temperature=1.0):
        super().__init__()
        if dim % int(num_heads) != 0:
            raise ValueError(f"TSPStateInductionBlock requires dim % num_heads == 0, got dim={dim}, num_heads={num_heads}.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.temperature = float(temperature)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.norm = nn.LayerNorm(self.dim)

    def _reshape_to_heads(self, x):
        # [N, C] -> [H, N, Dh]
        n, c = x.shape
        d_head = c // self.num_heads
        return x.view(n, self.num_heads, d_head).permute(1, 0, 2).contiguous()

    def _reshape_from_heads(self, x):
        # [H, N, Dh] -> [N, C]
        h, n, d_head = x.shape
        return x.permute(1, 0, 2).contiguous().view(n, h * d_head)

    def forward(self, query_slots, evidence_tokens, return_attn=False):
        # query_slots: [M, C]
        # evidence_tokens: [N, C]
        if evidence_tokens is None or evidence_tokens.shape[0] == 0:
            evidence_tokens = query_slots.new_zeros((1, query_slots.shape[-1]))
        input_dtype = query_slots.dtype
        q = self.q_proj(query_slots.to(dtype=self.q_proj.weight.dtype))
        k = self.k_proj(evidence_tokens.to(dtype=self.k_proj.weight.dtype))
        v = self.v_proj(evidence_tokens.to(dtype=self.v_proj.weight.dtype))

        qh = self._reshape_to_heads(q).float()
        kh = self._reshape_to_heads(k).float()
        vh = self._reshape_to_heads(v).float()
        d_head = max(int(qh.shape[-1]), 1)
        attn_logits = torch.matmul(qh, kh.transpose(-2, -1)) / (float(d_head) ** 0.5)
        temperature = max(float(self.temperature), 1e-6)
        attn_logits = attn_logits / temperature
        attn = F.softmax(attn_logits, dim=-1)
        if self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, vh)
        out = self._reshape_from_heads(out)
        out = out.to(dtype=self.out_proj.weight.dtype)
        out = self.out_proj(out)
        updated = self.norm(query_slots.to(dtype=out.dtype) + out)
        updated = updated.to(dtype=input_dtype)
        if return_attn:
            attn_weights_mean = attn.mean(dim=0)
            return updated, attn_weights_mean
        return updated, None


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        self._set_dvc_config(
            dvc_enable=getattr(config, "dvc_enable", False),
            dvc_zero_perturbation=getattr(config, "dvc_zero_perturbation", True),
            dvc_phase_schedule=getattr(config, "dvc_phase_schedule", "step"),
            dvc_warmup_ratio=getattr(config, "dvc_warmup_ratio", 0.1),
            dvc_entity_ratio=getattr(config, "dvc_entity_ratio", 0.7),
            dvc_layout_cells_per_axis=getattr(config, "dvc_layout_cells_per_axis", 6),
            dvc_role_embed=getattr(config, "dvc_role_embed", True),
            dvc_use_layout_adapter=getattr(config, "dvc_use_layout_adapter", True),
            dvc_use_role_embed=getattr(config, "dvc_use_role_embed", getattr(config, "dvc_role_embed", True)),
            dvc_use_layout_mix_gate=getattr(config, "dvc_use_layout_mix_gate", True),
            dvc_use_tail_role_heads=getattr(config, "dvc_use_tail_role_heads", True),
            dvc_use_slotwise_layout_mix=getattr(config, "dvc_use_slotwise_layout_mix", True),
            dvc_warmup_lr_scale=getattr(config, "dvc_warmup_lr_scale", 0.1),
            dvc_joint_lr_scale=getattr(config, "dvc_joint_lr_scale", 1.0),
            dvc_role_embed_zero_init=getattr(config, "dvc_role_embed_zero_init", True),
            dvc_variant=getattr(config, "dvc_variant", "legacy"),
            dvc_second_token_semantics=getattr(config, "dvc_second_token_semantics", "frame_anchor"),
            tsp_state_ent_slots=getattr(config, "tsp_state_ent_slots", getattr(config, "tsp_state_obj_slots", 4)),
            tsp_state_frame_slots=getattr(config, "tsp_state_frame_slots", getattr(config, "tsp_state_sup_slots", 4)),
            tsp_state_rel_slots=getattr(config, "tsp_state_rel_slots", 5),
            tsp_state_num_heads=getattr(config, "tsp_state_num_heads", 4),
            tsp_state_dropout=getattr(config, "tsp_state_dropout", 0.0),
            tsp_rel_seg_residual=getattr(config, "tsp_rel_seg_residual", getattr(config, "tsp_seg_query_residual", True)),
            tsp_rel_seg_alpha_init=getattr(config, "tsp_rel_seg_alpha_init", getattr(config, "tsp_seg_query_alpha_init", 0.0)),
            tsp_loss_sep=getattr(config, "tsp_loss_sep", 0.01),
            tsp_loss_div=getattr(config, "tsp_loss_div", 0.01),
            tsp_use_state_role_embed=getattr(config, "tsp_use_state_role_embed", True),
            tsp_use_ref_objects=getattr(config, "tsp_use_ref_objects", True),
            tsp_use_vertical_anchors=getattr(config, "tsp_use_vertical_anchors", True),
            tsp_ref_raw_tokens=getattr(config, "tsp_ref_raw_tokens", 8),
            tsp_struct_raw_tokens=getattr(config, "tsp_struct_raw_tokens", 8),
            tsp_debug_diversity=getattr(config, "tsp_debug_diversity", False),
            tsp_debug_diversity_print_once=getattr(config, "tsp_debug_diversity_print_once", True),
            tsp_use_slot_identity=getattr(config, "tsp_use_slot_identity", True),
            tsp_use_slot_competition=getattr(config, "tsp_use_slot_competition", True),
            tsp_rel_use_raw_evidence=getattr(config, "tsp_rel_use_raw_evidence", True),
            tsp_loss_attn_overlap=getattr(config, "tsp_loss_attn_overlap", 0.05),
            tsp_slot_identity_scale=getattr(config, "tsp_slot_identity_scale", 1.0),
            tsp_state_update_residual=getattr(config, "tsp_state_update_residual", True),
            tsp_state_update_alpha_init=getattr(config, "tsp_state_update_alpha_init", 1.0),
            tsp_slot_attention_temperature=getattr(config, "tsp_slot_attention_temperature", 1.0),
            tsp_rel_ref_raw_slots_cap=getattr(config, "tsp_rel_ref_raw_slots_cap", 8),
            tsp_rel_str_raw_slots_cap=getattr(config, "tsp_rel_str_raw_slots_cap", 8),
            tsp_rel_raw_bias=getattr(config, "tsp_rel_raw_bias", 2.0),
            tsp_rel_state_bias=getattr(config, "tsp_rel_state_bias", 0.5),
            tsp_loss_rel_raw_share=getattr(config, "tsp_loss_rel_raw_share", 0.05),
            tsp_rel_raw_share_target=getattr(config, "tsp_rel_raw_share_target", 0.65),
            tsp_rel_refine_enable=getattr(config, "tsp_rel_refine_enable", True),
            tsp_rel_refine_alpha_init=getattr(config, "tsp_rel_refine_alpha_init", 0.1),
            tsp_rel_refine_use_residual=getattr(config, "tsp_rel_refine_use_residual", True),
            tsp_rel_norm_control=getattr(config, "tsp_rel_norm_control", True),
            tsp_rel_norm_target_ratio=getattr(config, "tsp_rel_norm_target_ratio", 1.0),
            tsp_rel_norm_eps=getattr(config, "tsp_rel_norm_eps", 1e-6),
            tsp_sup_alpha_init=getattr(config, "tsp_sup_alpha_init", 0.5),
            tsp_sup_update_clamp=getattr(config, "tsp_sup_update_clamp", True),
            tsp_sup_update_max_scale=getattr(config, "tsp_sup_update_max_scale", 1.0),
            ssc_state_ent_slots=getattr(config, "ssc_state_ent_slots", 8),
            ssc_state_frame_slots=getattr(config, "ssc_state_frame_slots", 8),
            ssc_state_rel_slots=getattr(config, "ssc_state_rel_slots", 8),
            ssc_entity_candidate_topk=getattr(config, "ssc_entity_candidate_topk", 128),
            ssc_detail_tokens=getattr(config, "ssc_detail_tokens", -1),
            ssc_seg_alpha_init=getattr(config, "ssc_seg_alpha_init", 0.0),
            ssc_frame_cov_loss_weight=getattr(config, "ssc_frame_cov_loss_weight", 0.05),
            ssc_rel_boundary_loss_weight=getattr(config, "ssc_rel_boundary_loss_weight", 0.05),
            ssc_rel_region_loss_weight=getattr(config, "ssc_rel_region_loss_weight", 0.05),
            ssc_ent_sem_loss_weight=getattr(config, "ssc_ent_sem_loss_weight", 0.05),
            ssc_detail_dropout_rate=getattr(config, "ssc_detail_dropout_rate", 0.2),
            ssc_ablation_mode=getattr(config, "ssc_ablation_mode", "none"),
            ssc_role_analysis_variant=getattr(config, "ssc_role_analysis_variant", "none"),
        )

        if hasattr(config, "mm_vision_tower"):
            if config.mm_vision_tower is not None:
                self.vision_tower = build_vision_tower(config, delay_load=True)
                self.mm_projector = build_vision_projector(config)

                if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                    self.image_newline = nn.Parameter(
                        torch.empty(config.hidden_size, dtype=self.dtype)
                    )
        
        if hasattr(config, "mm_pointcloud_tower"):
            self.pointcloud_tower = build_pointcloud_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            pc_hidden_size = getattr(config, "pc_hidden_size", getattr(config, "mm_hidden_size", 1024))
            if getattr(config, "tsp_seg_hidden_size", None) is None:
                seg_dim = None
                try:
                    seg_dim = int(self.get_pointcloud_tower().seg_query_dim)
                except Exception:
                    seg_dim = None
                self.config.tsp_seg_hidden_size = seg_dim
                self.config.ssc_seg_hidden_size = seg_dim
            if not bool(getattr(self.config, "tsp_enable", False)):
                self._maybe_initialize_pc_dvc_modules(pc_hidden_size)
            self._maybe_initialize_pc_tsp_modules(pc_hidden_size)
            self._maybe_initialize_pc_ssc_modules(pc_hidden_size)

        
        if hasattr(config, "mm_inst_prompt_encoder"):
            if not config.mm_inst_prompt_encoder == "shared_projector":
                self.inst_prompt_encoder = build_inst_prompt_encoder(config)
        
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_pointcloud_tower(self):
        pointcloud_tower = getattr(self, 'pointcloud_tower', None)
        if type(pointcloud_tower) is list:
            pointcloud_tower = pointcloud_tower[0]
        return pointcloud_tower
    
    def get_mask_decoder(self):
        pointcloud_tower = self.get_pointcloud_tower()
        mask_decoder = pointcloud_tower.mask_decoder
        return mask_decoder

    def get_hidden_seg_fc(self):
        pointcloud_tower = self.get_pointcloud_tower()
        hidden_seg_fc = pointcloud_tower.hidden_seg_fc
        return hidden_seg_fc

    def get_seg_criteria(self):
        pointcloud_tower = self.get_pointcloud_tower()
        seg_criteria = pointcloud_tower.seg_criteria
        return seg_criteria

    def get_prompt_encoder(self):
        prompt_encoder = getattr(self, 'prompt_encoder', None)
        return prompt_encoder

    def get_inst_prompt_encoder(self):
        inst_prompt_encoder = getattr(self, 'inst_prompt_encoder', None)
        return inst_prompt_encoder

    def _set_dvc_config(
        self,
        dvc_enable=False,
        dvc_zero_perturbation=True,
        dvc_phase_schedule="step",
        dvc_warmup_ratio=0.1,
        dvc_entity_ratio=0.7,
        dvc_layout_cells_per_axis=6,
        dvc_role_embed=True,
        dvc_use_layout_adapter=True,
        dvc_use_role_embed=True,
        dvc_use_layout_mix_gate=True,
        dvc_use_tail_role_heads=True,
        dvc_use_slotwise_layout_mix=True,
        dvc_warmup_lr_scale=0.1,
        dvc_joint_lr_scale=1.0,
        dvc_role_embed_zero_init=True,
        dvc_variant="legacy",
        dvc_second_token_semantics="frame_anchor",
        tsp_state_ent_slots=4,
        tsp_state_frame_slots=4,
        tsp_state_rel_slots=4,
        tsp_state_num_heads=4,
        tsp_state_dropout=0.0,
        tsp_rel_seg_residual=True,
        tsp_rel_seg_alpha_init=0.0,
        # Compatibility-only options (kept for legacy checkpoints/CLI)
        tsp_state_obj_slots=5,
        tsp_state_sup_slots=5,
        tsp_seg_query_residual=True,
        tsp_seg_query_alpha_init=0.0,
        tsp_loss_sep=0.01,
        tsp_loss_div=0.01,
        tsp_use_state_role_embed=True,
        tsp_use_ref_objects=True,
        tsp_use_vertical_anchors=True,
        tsp_ref_raw_tokens=8,
        tsp_struct_raw_tokens=8,
        tsp_debug_diversity=False,
        tsp_debug_diversity_print_once=True,
        tsp_use_slot_identity=True,
        tsp_use_slot_competition=True,
        tsp_rel_use_raw_evidence=True,
        tsp_loss_attn_overlap=0.05,
        tsp_slot_identity_scale=1.0,
        tsp_state_update_residual=True,
        tsp_state_update_alpha_init=1.0,
        tsp_slot_attention_temperature=1.0,
        tsp_rel_ref_raw_slots_cap=8,
        tsp_rel_str_raw_slots_cap=8,
        tsp_rel_raw_bias=2.0,
        tsp_rel_state_bias=0.5,
        tsp_loss_rel_raw_share=0.05,
        tsp_rel_raw_share_target=0.65,
        tsp_rel_refine_enable=True,
        tsp_rel_refine_alpha_init=0.1,
        tsp_rel_refine_use_residual=True,
        tsp_rel_norm_control=True,
        tsp_rel_norm_target_ratio=1.0,
        tsp_rel_norm_eps=1e-6,
        tsp_sup_alpha_init=0.5,
        tsp_sup_update_clamp=True,
        tsp_sup_update_max_scale=1.0,
        ssc_state_ent_slots=8,
        ssc_state_frame_slots=8,
        ssc_state_rel_slots=8,
        ssc_scene_summary_token_count=1,
        ssc_entity_candidate_topk=128,
        ssc_detail_tokens=-1,
        ssc_seg_alpha_init=0.0,
        ssc_frame_cov_loss_weight=0.05,
        ssc_rel_boundary_loss_weight=0.05,
        ssc_rel_region_loss_weight=0.05,
        ssc_ent_sem_loss_weight=0.05,
        ssc_detail_dropout_rate=0.2,
        ssc_ablation_mode="none",
        ssc_role_analysis_variant="none",
    ):
        dvc_variant = str(dvc_variant)
        second_token_semantics = str(dvc_second_token_semantics)
        if dvc_variant in {"tsp3d", "ssc3d"} and second_token_semantics in {"generic_layout", "relation_anchor", "hybrid_ra", "gta_lite"}:
            second_token_semantics = "frame_anchor"
        tsp_enable = dvc_variant == "tsp3d"
        ssc_enable = dvc_variant == "ssc3d"
        if tsp_enable:
            dvc_use_layout_adapter = False
            dvc_use_layout_mix_gate = False
            dvc_use_tail_role_heads = False
            dvc_use_slotwise_layout_mix = False
            dvc_zero_perturbation = False
            dvc_phase_schedule = "none"
            self.config.dvc_use_layout_constructor = False
            self.config.dvc_use_query_conditioned_constructor = False
            self.config.dvc_use_dynamic_budget = False
            self.config.dvc_use_object_core_preserving = False
        if ssc_enable:
            dvc_use_layout_adapter = False
            dvc_use_layout_mix_gate = False
            dvc_use_tail_role_heads = False
            dvc_use_slotwise_layout_mix = False
            dvc_zero_perturbation = False
            dvc_phase_schedule = "none"
            self.config.dvc_use_layout_constructor = False
            self.config.dvc_use_query_conditioned_constructor = False
            self.config.dvc_use_dynamic_budget = False
            self.config.dvc_use_object_core_preserving = False

        self.config.dvc_enable = dvc_enable
        self.config.dvc_second_token_semantics = second_token_semantics
        self.config.dvc_zero_perturbation = dvc_zero_perturbation
        self.config.dvc_phase_schedule = dvc_phase_schedule
        self.config.dvc_warmup_ratio = dvc_warmup_ratio
        self.config.dvc_entity_ratio = dvc_entity_ratio
        self.config.dvc_layout_cells_per_axis = dvc_layout_cells_per_axis
        self.config.dvc_role_embed = dvc_role_embed
        self.config.dvc_use_layout_adapter = dvc_use_layout_adapter
        self.config.dvc_use_role_embed = dvc_use_role_embed
        self.config.dvc_use_layout_mix_gate = dvc_use_layout_mix_gate
        self.config.dvc_use_tail_role_heads = dvc_use_tail_role_heads
        self.config.dvc_use_slotwise_layout_mix = dvc_use_slotwise_layout_mix
        self.config.dvc_warmup_lr_scale = dvc_warmup_lr_scale
        self.config.dvc_joint_lr_scale = dvc_joint_lr_scale
        self.config.dvc_role_embed_zero_init = dvc_role_embed_zero_init
        self.config.dvc_variant = dvc_variant
        self.config.tsp_enable = bool(tsp_enable)
        self.config.ssc_enable = bool(ssc_enable)
        self.config.tsp_state_ent_slots = int(tsp_state_ent_slots)
        self.config.tsp_state_frame_slots = int(tsp_state_frame_slots)
        self.config.tsp_state_rel_slots = int(tsp_state_rel_slots)
        self.config.tsp_state_num_heads = int(tsp_state_num_heads)
        self.config.tsp_state_dropout = float(tsp_state_dropout)
        self.config.tsp_rel_seg_residual = bool(tsp_rel_seg_residual)
        self.config.tsp_rel_seg_alpha_init = float(tsp_rel_seg_alpha_init)
        # Backward-compatible aliases.
        self.config.tsp_state_obj_slots = int(tsp_state_ent_slots)
        self.config.tsp_state_sup_slots = int(tsp_state_frame_slots)
        self.config.tsp_seg_query_residual = bool(tsp_rel_seg_residual)
        self.config.tsp_seg_query_alpha_init = float(tsp_rel_seg_alpha_init)

        # Legacy/debug knobs are kept in config for parser/checkpoint compatibility.
        self.config.tsp_loss_sep = float(tsp_loss_sep)
        self.config.tsp_loss_div = float(tsp_loss_div)
        self.config.tsp_use_state_role_embed = bool(tsp_use_state_role_embed)
        self.config.tsp_use_ref_objects = bool(tsp_use_ref_objects)
        self.config.tsp_use_vertical_anchors = bool(tsp_use_vertical_anchors)
        self.config.tsp_ref_raw_tokens = int(tsp_ref_raw_tokens)
        self.config.tsp_struct_raw_tokens = int(tsp_struct_raw_tokens)
        self.config.tsp_debug_diversity = bool(tsp_debug_diversity)
        self.config.tsp_debug_diversity_print_once = bool(tsp_debug_diversity_print_once)
        self.config.tsp_use_slot_identity = bool(tsp_use_slot_identity)
        self.config.tsp_use_slot_competition = bool(tsp_use_slot_competition)
        self.config.tsp_rel_use_raw_evidence = bool(tsp_rel_use_raw_evidence)
        self.config.tsp_loss_attn_overlap = float(tsp_loss_attn_overlap)
        self.config.tsp_slot_identity_scale = float(tsp_slot_identity_scale)
        self.config.tsp_state_update_residual = bool(tsp_state_update_residual)
        self.config.tsp_state_update_alpha_init = float(tsp_state_update_alpha_init)
        self.config.tsp_slot_attention_temperature = float(tsp_slot_attention_temperature)
        self.config.tsp_rel_ref_raw_slots_cap = int(tsp_rel_ref_raw_slots_cap)
        self.config.tsp_rel_str_raw_slots_cap = int(tsp_rel_str_raw_slots_cap)
        self.config.tsp_rel_raw_bias = float(tsp_rel_raw_bias)
        self.config.tsp_rel_state_bias = float(tsp_rel_state_bias)
        self.config.tsp_loss_rel_raw_share = float(tsp_loss_rel_raw_share)
        self.config.tsp_rel_raw_share_target = float(tsp_rel_raw_share_target)
        self.config.tsp_rel_refine_enable = bool(tsp_rel_refine_enable)
        self.config.tsp_rel_refine_alpha_init = float(tsp_rel_refine_alpha_init)
        self.config.tsp_rel_refine_use_residual = bool(tsp_rel_refine_use_residual)
        self.config.tsp_rel_norm_control = bool(tsp_rel_norm_control)
        self.config.tsp_rel_norm_target_ratio = float(tsp_rel_norm_target_ratio)
        self.config.tsp_rel_norm_eps = float(tsp_rel_norm_eps)
        self.config.tsp_sup_alpha_init = float(tsp_sup_alpha_init)
        self.config.tsp_sup_update_clamp = bool(tsp_sup_update_clamp)
        self.config.tsp_sup_update_max_scale = float(tsp_sup_update_max_scale)
        self.config.tsp_seg_hidden_size = getattr(self.config, "tsp_seg_hidden_size", None)
        self.config.ssc_state_ent_slots = int(ssc_state_ent_slots)
        self.config.ssc_state_frame_slots = int(ssc_state_frame_slots)
        self.config.ssc_state_rel_slots = int(ssc_state_rel_slots)
        self.config.ssc_scene_summary_token_count = int(ssc_scene_summary_token_count)
        self.config.ssc_entity_candidate_topk = int(ssc_entity_candidate_topk)
        self.config.ssc_detail_tokens = int(ssc_detail_tokens)
        self.config.ssc_seg_alpha_init = float(ssc_seg_alpha_init)
        self.config.ssc_frame_cov_loss_weight = float(ssc_frame_cov_loss_weight)
        self.config.ssc_rel_boundary_loss_weight = float(ssc_rel_boundary_loss_weight)
        self.config.ssc_rel_region_loss_weight = float(ssc_rel_region_loss_weight)
        self.config.ssc_ent_sem_loss_weight = float(ssc_ent_sem_loss_weight)
        self.config.ssc_detail_dropout_rate = float(ssc_detail_dropout_rate)
        self.config.ssc_ablation_mode = normalize_ssc_ablation_mode(ssc_ablation_mode)
        self.config.ssc_role_analysis_variant = normalize_ssc_role_analysis_variant(ssc_role_analysis_variant)
        self.config.ssc_seg_hidden_size = getattr(self.config, "ssc_seg_hidden_size", None)

        self.config.current_dvc_phase = "none" if dvc_phase_schedule == "none" else getattr(self.config, "current_dvc_phase", "dvc_warmup")
        self.config.current_dvc_gate_scale = getattr(self.config, "current_dvc_gate_scale", 0.0 if dvc_zero_perturbation else 1.0)

    def _build_gated_residual_adapter(self, pc_hidden_size):
        return GatedResidualAdapter(pc_hidden_size)

    def _maybe_initialize_pc_dvc_modules(self, pc_hidden_size):
        if bool(getattr(self.config, "tsp_enable", False)) or str(getattr(self.config, "dvc_variant", "legacy")) == "ssc3d":
            return

        if self.config.dvc_enable and getattr(self, "pc_layout_adapter", None) is None:
            self.pc_layout_adapter = self._build_gated_residual_adapter(pc_hidden_size)

        if self.config.dvc_enable and getattr(self, "pc_entity_tail_role_head", None) is None:
            self.pc_entity_tail_role_head = self._build_gated_residual_adapter(self.config.hidden_size)

        if self.config.dvc_enable and getattr(self, "pc_layout_role_head", None) is None:
            self.pc_layout_role_head = self._build_gated_residual_adapter(self.config.hidden_size)

        if self.config.dvc_enable and getattr(self, "pc_role_embed", None) is None:
            self.pc_role_embed = nn.Parameter(torch.zeros(2, self.config.hidden_size))
            if self.config.dvc_role_embed_zero_init:
                nn.init.zeros_(self.pc_role_embed)
            else:
                nn.init.normal_(self.pc_role_embed, std=0.02)

        if self.config.dvc_enable and getattr(self, "pc_layout_mix_gate", None) is None:
            self.pc_layout_mix_gate = nn.Parameter(torch.zeros(1))

        if self.config.dvc_enable and getattr(self, "pc_layout_slot_gate", None) is None:
            self.pc_layout_slot_gate = SlotWiseMixGate(self.config.hidden_size)

    def _maybe_initialize_pc_tsp_modules(self, pc_hidden_size):
        if not bool(getattr(self.config, "tsp_enable", False)):
            return

        seg_hidden_size = getattr(self.config, "tsp_seg_hidden_size", None)
        if seg_hidden_size is None:
            pointcloud_tower = self.get_pointcloud_tower()
            if pointcloud_tower is not None and hasattr(pointcloud_tower, "seg_query_dim"):
                try:
                    seg_hidden_size = int(pointcloud_tower.seg_query_dim)
                except Exception:
                    seg_hidden_size = None
        if seg_hidden_size is None:
            return
        seg_hidden_size = int(seg_hidden_size)
        self.config.tsp_seg_hidden_size = seg_hidden_size

        state_heads = max(int(getattr(self.config, "tsp_state_num_heads", 4)), 1)
        if pc_hidden_size % state_heads != 0:
            for candidate in range(state_heads, 0, -1):
                if pc_hidden_size % candidate == 0:
                    state_heads = candidate
                    break
            self.config.tsp_state_num_heads = int(state_heads)

        attn_dropout = float(getattr(self.config, "tsp_state_dropout", 0.0))
        m_ent = int(getattr(self.config, "tsp_state_ent_slots", getattr(self.config, "tsp_state_obj_slots", 4)))
        m_frame = int(getattr(self.config, "tsp_state_frame_slots", getattr(self.config, "tsp_state_sup_slots", 4)))
        m_rel = int(getattr(self.config, "tsp_state_rel_slots", 4))
        slot_temperature = 1.0

        def _init_or_resize_param(name, shape, init_std=0.02, init_value=None):
            param = getattr(self, name, None)
            if shape == () or shape == tuple():
                shape = (1,)
            if (
                param is None
                or not isinstance(param, nn.Parameter)
                or tuple(param.shape) != tuple(shape)
            ):
                if init_value is None:
                    new_param = nn.Parameter(torch.randn(*shape) * init_std)
                else:
                    new_param = nn.Parameter(torch.full(shape, float(init_value), dtype=torch.float32))
                setattr(self, name, new_param)

        _init_or_resize_param("pc_state_query_ent", (m_ent, pc_hidden_size), init_std=0.02)
        _init_or_resize_param("pc_state_query_frame", (m_frame, pc_hidden_size), init_std=0.02)
        _init_or_resize_param("pc_state_query_rel", (m_rel, pc_hidden_size), init_std=0.02)
        _init_or_resize_param(
            "pc_rel_seg_alpha",
            (1,),
            init_value=float(getattr(self.config, "tsp_rel_seg_alpha_init", 0.0)),
        )
        if (
            self.config.dvc_enable
            and (
                getattr(self, "pc_role_embed", None) is None
                or not isinstance(self.pc_role_embed, nn.Parameter)
                or tuple(self.pc_role_embed.shape) != (2, self.config.hidden_size)
            )
        ):
            self.pc_role_embed = nn.Parameter(torch.zeros(2, self.config.hidden_size))
            if self.config.dvc_role_embed_zero_init:
                nn.init.zeros_(self.pc_role_embed)
            else:
                nn.init.normal_(self.pc_role_embed, std=0.02)

        def _build_state_block(existing):
            if (
                existing is None
                or not isinstance(existing, TSPStateInductionBlock)
                or int(getattr(existing, "dim", -1)) != int(pc_hidden_size)
                or int(getattr(existing, "num_heads", -1)) != int(state_heads)
            ):
                return TSPStateInductionBlock(
                    dim=pc_hidden_size,
                    num_heads=state_heads,
                    dropout=attn_dropout,
                    temperature=slot_temperature,
                )
            existing.temperature = float(slot_temperature)
            existing.dropout = float(attn_dropout)
            return existing

        self.pc_state_ent_block = _build_state_block(getattr(self, "pc_state_ent_block", None))
        self.pc_state_frame_block = _build_state_block(getattr(self, "pc_state_frame_block", None))
        self.pc_state_rel_block = _build_state_block(getattr(self, "pc_state_rel_block", None))
        if (
            getattr(self, "pc_rel_to_seg", None) is None
            or self.pc_rel_to_seg.in_features != pc_hidden_size
            or self.pc_rel_to_seg.out_features != seg_hidden_size
        ):
            self.pc_rel_to_seg = nn.Linear(pc_hidden_size, seg_hidden_size)

    def _maybe_initialize_pc_ssc_modules(self, pc_hidden_size):
        if str(getattr(self.config, "dvc_variant", "legacy")) != "ssc3d":
            return

        seg_hidden_size = getattr(self.config, "ssc_seg_hidden_size", None)
        if seg_hidden_size is None:
            pointcloud_tower = self.get_pointcloud_tower()
            if pointcloud_tower is not None and hasattr(pointcloud_tower, "seg_query_dim"):
                try:
                    seg_hidden_size = int(pointcloud_tower.seg_query_dim)
                except Exception:
                    seg_hidden_size = None
        if seg_hidden_size is None:
            return
        seg_hidden_size = int(seg_hidden_size)
        self.config.ssc_seg_hidden_size = seg_hidden_size

        state_heads = max(int(getattr(self.config, "tsp_state_num_heads", 4)), 1)
        if pc_hidden_size % state_heads != 0:
            for candidate in range(state_heads, 0, -1):
                if pc_hidden_size % candidate == 0:
                    state_heads = candidate
                    break
            self.config.tsp_state_num_heads = int(state_heads)

        attn_dropout = float(getattr(self.config, "tsp_state_dropout", 0.0))
        m_ent = int(getattr(self.config, "ssc_state_ent_slots", 8))
        m_frame = int(getattr(self.config, "ssc_state_frame_slots", 8))
        m_rel = int(getattr(self.config, "ssc_state_rel_slots", 8))
        m_scene = max(int(getattr(self.config, "ssc_scene_summary_token_count", 1)), 1)
        m_generic = max(m_ent + m_frame + m_rel + m_scene, 1)

        if (
            getattr(self, "pc_ssc_entity_primitive_head", None) is None
            or not isinstance(self.pc_ssc_entity_primitive_head, nn.Sequential)
            or len(self.pc_ssc_entity_primitive_head) != 4
            or self.pc_ssc_entity_primitive_head[1].in_features != pc_hidden_size
            or self.pc_ssc_entity_primitive_head[1].out_features != pc_hidden_size
            or self.pc_ssc_entity_primitive_head[3].out_features != 1
        ):
            self.pc_ssc_entity_primitive_head = nn.Sequential(
                nn.LayerNorm(pc_hidden_size),
                nn.Linear(pc_hidden_size, pc_hidden_size),
                nn.GELU(),
                nn.Linear(pc_hidden_size, 1),
            )
        if (
            getattr(self, "pc_ssc_frame_object_primitive_head", None) is None
            or not isinstance(self.pc_ssc_frame_object_primitive_head, nn.Sequential)
            or len(self.pc_ssc_frame_object_primitive_head) != 4
            or self.pc_ssc_frame_object_primitive_head[1].in_features != pc_hidden_size
            or self.pc_ssc_frame_object_primitive_head[1].out_features != pc_hidden_size
            or self.pc_ssc_frame_object_primitive_head[3].out_features != 1
        ):
            self.pc_ssc_frame_object_primitive_head = nn.Sequential(
                nn.LayerNorm(pc_hidden_size),
                nn.Linear(pc_hidden_size, pc_hidden_size),
                nn.GELU(),
                nn.Linear(pc_hidden_size, 1),
            )

        def _init_or_resize_param(name, shape, init_std=0.02, init_value=None):
            param = getattr(self, name, None)
            if shape == () or shape == tuple():
                shape = (1,)
            if (
                param is None
                or not isinstance(param, nn.Parameter)
                or tuple(param.shape) != tuple(shape)
            ):
                if init_value is None:
                    new_param = nn.Parameter(torch.randn(*shape) * init_std)
                else:
                    new_param = nn.Parameter(torch.full(shape, float(init_value), dtype=torch.float32))
                setattr(self, name, new_param)

        _init_or_resize_param("pc_ssc_state_query_ent", (m_ent, pc_hidden_size), init_std=0.02)
        _init_or_resize_param("pc_ssc_state_query_frame", (m_frame, pc_hidden_size), init_std=0.02)
        _init_or_resize_param("pc_ssc_state_query_rel", (m_rel, pc_hidden_size), init_std=0.02)
        _init_or_resize_param("pc_ssc_state_query_generic", (m_generic, pc_hidden_size), init_std=0.02)
        # SSC3D serialized role order: detail, E, F, R, S_scene.
        _init_or_resize_param("pc_ssc_token_role_embed", (5, self.config.hidden_size), init_value=0.0)
        _init_or_resize_param("pc_ssc_generic_token_embed", (1, self.config.hidden_size), init_value=0.0)
        _init_or_resize_param(
            "pc_ssc_seg_alpha",
            (1,),
            init_value=float(getattr(self.config, "ssc_seg_alpha_init", 0.0)),
        )

        def _build_state_block(existing):
            if (
                existing is None
                or not isinstance(existing, TSPStateInductionBlock)
                or int(getattr(existing, "dim", -1)) != int(pc_hidden_size)
                or int(getattr(existing, "num_heads", -1)) != int(state_heads)
            ):
                return TSPStateInductionBlock(
                    dim=pc_hidden_size,
                    num_heads=state_heads,
                    dropout=attn_dropout,
                    temperature=1.0,
                )
            existing.temperature = 1.0
            existing.dropout = float(attn_dropout)
            return existing

        self.pc_ssc_state_ent_block = _build_state_block(getattr(self, "pc_ssc_state_ent_block", None))
        self.pc_ssc_state_frame_block = _build_state_block(getattr(self, "pc_ssc_state_frame_block", None))
        self.pc_ssc_state_rel_block = _build_state_block(getattr(self, "pc_ssc_state_rel_block", None))
        self.pc_ssc_state_generic_block = _build_state_block(getattr(self, "pc_ssc_state_generic_block", None))

        if (
            getattr(self, "pc_ssc_seg_reader", None) is None
            or self.pc_ssc_seg_reader.in_features != 3 * pc_hidden_size
            or self.pc_ssc_seg_reader.out_features != seg_hidden_size
        ):
            self.pc_ssc_seg_reader = nn.Linear(3 * pc_hidden_size, seg_hidden_size)

        if (
            getattr(self, "pc_ssc_scene_proj", None) is None
            or self.pc_ssc_scene_proj.in_features != 3 * pc_hidden_size
            or self.pc_ssc_scene_proj.out_features != m_scene * pc_hidden_size
        ):
            self.pc_ssc_scene_proj = nn.Linear(3 * pc_hidden_size, m_scene * pc_hidden_size)

        if (
            getattr(self, "pc_ssc_ent_sem_proj", None) is None
            or not isinstance(self.pc_ssc_ent_sem_proj, nn.Sequential)
            or len(self.pc_ssc_ent_sem_proj) != 2
            or self.pc_ssc_ent_sem_proj[1].in_features != pc_hidden_size
            or self.pc_ssc_ent_sem_proj[1].out_features != pc_hidden_size
        ):
            self.pc_ssc_ent_sem_proj = nn.Sequential(
                nn.LayerNorm(pc_hidden_size),
                nn.Linear(pc_hidden_size, pc_hidden_size),
            )

        if (
            getattr(self, "pc_ssc_rel_boundary_head", None) is None
            or self.pc_ssc_rel_boundary_head.in_features != pc_hidden_size
            or self.pc_ssc_rel_boundary_head.out_features != 4
        ):
            self.pc_ssc_rel_boundary_head = nn.Linear(pc_hidden_size, 4)

        num_region_classes = max(
            int(getattr(self.config, "dvc_region_grid_x", 3)) * int(getattr(self.config, "dvc_region_grid_y", 3)),
            1,
        )
        if (
            getattr(self, "pc_ssc_rel_region_head", None) is None
            or self.pc_ssc_rel_region_head.in_features != pc_hidden_size
            or self.pc_ssc_rel_region_head.out_features != num_region_classes
        ):
            self.pc_ssc_rel_region_head = nn.Linear(pc_hidden_size, num_region_classes)

    def _resolve_ssc_ablation_layout(self, num_pc_tokens=None):
        if num_pc_tokens is None:
            num_pc_tokens = int(getattr(self.config, "num_pc_tokens", 0))
        return resolve_ssc_ablation_layout(
            num_pc_tokens=num_pc_tokens,
            ssc_state_ent_slots=getattr(self.config, "ssc_state_ent_slots", 8),
            ssc_state_frame_slots=getattr(self.config, "ssc_state_frame_slots", 8),
            ssc_state_rel_slots=getattr(self.config, "ssc_state_rel_slots", 8),
            ssc_scene_summary_token_count=getattr(self.config, "ssc_scene_summary_token_count", 1),
            ssc_detail_tokens=getattr(self.config, "ssc_detail_tokens", -1),
            ssc_ablation_mode=getattr(self.config, "ssc_ablation_mode", "none"),
        )

    def _resolve_ssc_role_analysis_settings(self, num_pc_tokens=None):
        ssc_layout = self._resolve_ssc_ablation_layout(num_pc_tokens=num_pc_tokens)
        return resolve_ssc_role_analysis_settings(
            getattr(self.config, "ssc_role_analysis_variant", "none"),
            ssc_layout["original_state_total"],
        )

    def synthesize_tsp_seg_queries(self, seg_embeds, mask_input_dict):
        if not bool(getattr(self.config, "tsp_enable", False)):
            return seg_embeds

        state_bank = mask_input_dict.get("tsp_state_bank", {})
        rel_bank = state_bank.get("S_rel", [])
        new_seg_embeds = []
        alpha = self.pc_rel_seg_alpha
        use_residual = bool(getattr(self.config, "tsp_rel_seg_residual", True))

        for scene_idx, q_base in enumerate(seg_embeds):
            if q_base.shape[0] == 0:
                new_seg_embeds.append(q_base)
                continue
            seg_hidden_size = int(getattr(self.config, "tsp_seg_hidden_size", q_base.shape[-1]))
            if int(q_base.shape[-1]) != seg_hidden_size:
                raise RuntimeError(
                    f"TSP3D seg query dim mismatch: q_base_dim={int(q_base.shape[-1])}, "
                    f"tsp_seg_hidden_size={seg_hidden_size}."
                )
            s_rel_raw = rel_bank[scene_idx] if scene_idx < len(rel_bank) else None

            if s_rel_raw is None or int(s_rel_raw.shape[0]) == 0:
                q_final = q_base
            else:
                pooled_rel = s_rel_raw.mean(dim=0, keepdim=True).to(device=q_base.device)
                rel_bias = self.pc_rel_to_seg(pooled_rel.to(dtype=self.pc_rel_to_seg.weight.dtype))
                rel_bias = rel_bias.to(dtype=q_base.dtype, device=q_base.device)
                alpha_scene = alpha.to(device=q_base.device, dtype=q_base.dtype).view(1)
                rel_delta = alpha_scene * rel_bias.expand_as(q_base)
                if use_residual:
                    q_final = q_base + rel_delta
                else:
                    q_final = rel_delta
            new_seg_embeds.append(q_final)
        return new_seg_embeds

    def synthesize_ssc_seg_queries(self, seg_embeds, mask_input_dict):
        if str(getattr(self.config, "dvc_variant", "legacy")) != "ssc3d":
            return seg_embeds
        ssc_role_analysis = self._resolve_ssc_role_analysis_settings()
        if not bool(ssc_role_analysis["seg_role_bias_enabled"]):
            return seg_embeds

        state_bank = mask_input_dict.get("ssc_state_bank", {})
        ent_bank = state_bank.get("S_ent", [])
        frame_bank = state_bank.get("S_frame", [])
        rel_bank = state_bank.get("S_rel", [])
        new_seg_embeds = []
        seg_query_debug = []
        alpha = self.pc_ssc_seg_alpha

        for scene_idx, q_base in enumerate(seg_embeds):
            if q_base.shape[0] == 0:
                new_seg_embeds.append(q_base)
                seg_query_debug.append({"scene_index": int(scene_idx), "applied": False, "reason": "empty_query"})
                continue
            seg_hidden_size = int(getattr(self.config, "ssc_seg_hidden_size", q_base.shape[-1]))
            if int(q_base.shape[-1]) != seg_hidden_size:
                raise RuntimeError(
                    f"SSC3D seg query dim mismatch: q_base_dim={int(q_base.shape[-1])}, "
                    f"ssc_seg_hidden_size={seg_hidden_size}."
                )

            s_ent = ent_bank[scene_idx] if scene_idx < len(ent_bank) else None
            s_frame = frame_bank[scene_idx] if scene_idx < len(frame_bank) else None
            s_rel = rel_bank[scene_idx] if scene_idx < len(rel_bank) else None
            if (
                s_ent is None or int(s_ent.shape[0]) == 0
                or s_frame is None or int(s_frame.shape[0]) == 0
                or s_rel is None or int(s_rel.shape[0]) == 0
            ):
                new_seg_embeds.append(q_base)
                seg_query_debug.append({"scene_index": int(scene_idx), "applied": False, "reason": "missing_state"})
                continue

            pooled_ent = s_ent.mean(dim=0, keepdim=True)
            pooled_frame = s_frame.mean(dim=0, keepdim=True)
            pooled_rel = s_rel.mean(dim=0, keepdim=True)
            pooled_state = torch.cat([pooled_ent, pooled_frame, pooled_rel], dim=-1).to(device=q_base.device)
            seg_bias = self.pc_ssc_seg_reader(pooled_state.to(dtype=self.pc_ssc_seg_reader.weight.dtype))
            seg_bias = seg_bias.to(dtype=q_base.dtype, device=q_base.device)
            alpha_scene = alpha.to(device=q_base.device, dtype=q_base.dtype).view(1)
            q_final = q_base + alpha_scene * seg_bias.expand_as(q_base)
            seg_query_debug.append({
                "scene_index": int(scene_idx),
                "applied": True,
                "base_norm": float(q_base.float().norm().item()),
                "bias_norm": float((alpha_scene * seg_bias).float().norm().item()),
                "final_norm": float(q_final.float().norm().item()),
            })
            new_seg_embeds.append(q_final)
        mask_input_dict["ssc_seg_query_debug"] = seg_query_debug
        self._last_ssc_seg_query_debug = seg_query_debug
        return new_seg_embeds

    def _compute_ssc_frame_coverage_loss(self, attn_frame, frame_group_ids):
        if attn_frame is None:
            return None
        if not isinstance(frame_group_ids, (list, tuple)) or len(frame_group_ids) == 0:
            return attn_frame.float().sum() * 0.0

        attn = attn_frame.float()
        if attn.ndim > 2:
            attn = attn.mean(dim=0)
        if attn.ndim != 2 or int(attn.shape[-1]) == 0:
            return attn.sum() * 0.0
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        num_src = int(attn.shape[-1])
        valid_group_ids = [int(v) for v in frame_group_ids[:num_src] if int(v) >= 0]
        if len(valid_group_ids) == 0:
            return attn.sum() * 0.0

        group_tensor = torch.tensor(
            [int(v) for v in frame_group_ids[:num_src]],
            dtype=torch.long,
            device=attn.device,
        )
        losses = []
        for group_id in sorted(set(valid_group_ids)):
            group_mask = group_tensor == int(group_id)
            if not bool(group_mask.any()):
                continue
            group_coverage = attn[:, group_mask].sum(dim=-1)
            max_slot_coverage = group_coverage.max()
            losses.append(-torch.log(max_slot_coverage.clamp_min(1e-6)))
        if len(losses) == 0:
            return attn.sum() * 0.0
        return torch.stack(losses).mean()

    def _compute_ssc_relation_geo_loss(self, S_rel, entity_primitive_xyz, sp_xyz, entity_primitive_weights=None):
        if sp_xyz is None or int(sp_xyz.shape[0]) == 0:
            raise RuntimeError("SSC3D relation geometry loss requires valid sp_xyz.")

        if S_rel is None or int(S_rel.shape[0]) == 0:
            zero = sp_xyz.float().sum() * 0.0
            return zero, zero

        if entity_primitive_xyz is None or int(entity_primitive_xyz.shape[0]) == 0:
            zero = S_rel.float().sum() * 0.0
            return zero, zero

        if (
            entity_primitive_weights is not None
            and torch.is_tensor(entity_primitive_weights)
            and int(entity_primitive_weights.numel()) == int(entity_primitive_xyz.shape[0])
            and int(entity_primitive_weights.numel()) > 0
        ):
            w = entity_primitive_weights.float().to(device=entity_primitive_xyz.device).view(-1)
            w = w / w.sum().clamp_min(1e-6)
            centroid = (entity_primitive_xyz.float() * w.unsqueeze(-1)).sum(dim=0)
        else:
            centroid = entity_primitive_xyz.float().mean(dim=0)
        scene_xyz = sp_xyz.float()
        xyz_min = scene_xyz.min(dim=0)[0]
        xyz_max = scene_xyz.max(dim=0)[0]

        boundary_dist = torch.stack(
            [
                centroid[0] - xyz_min[0],
                xyz_max[0] - centroid[0],
                centroid[1] - xyz_min[1],
                xyz_max[1] - centroid[1],
            ],
            dim=0,
        )
        boundary_label = boundary_dist.argmin().view(1).to(device=S_rel.device, dtype=torch.long)

        grid_x = max(int(getattr(self.config, "dvc_region_grid_x", 3)), 1)
        grid_y = max(int(getattr(self.config, "dvc_region_grid_y", 3)), 1)
        xy_span = (xyz_max[:2] - xyz_min[:2]).clamp_min(1e-6)
        norm_xy = ((centroid[:2] - xyz_min[:2]) / xy_span).clamp(0.0, 1.0 - 1e-6)
        cell_x = torch.floor(norm_xy[0] * grid_x).long().clamp(0, grid_x - 1)
        cell_y = torch.floor(norm_xy[1] * grid_y).long().clamp(0, grid_y - 1)
        region_label = (cell_y * grid_x + cell_x).view(1).to(device=S_rel.device, dtype=torch.long)

        pooled_rel = S_rel.mean(dim=0, keepdim=True)
        boundary_logits = self.pc_ssc_rel_boundary_head(
            pooled_rel.to(dtype=self.pc_ssc_rel_boundary_head.weight.dtype)
        )
        region_logits = self.pc_ssc_rel_region_head(
            pooled_rel.to(dtype=self.pc_ssc_rel_region_head.weight.dtype)
        )
        loss_boundary = F.cross_entropy(boundary_logits.float(), boundary_label)
        loss_region = F.cross_entropy(region_logits.float(), region_label)
        return loss_boundary, loss_region

    def _compute_ssc_entity_semantic_loss(self, S_ent, entity_primitives, entity_primitive_weights):
        if S_ent is None or int(S_ent.shape[0]) == 0:
            zero = self.pc_ssc_ent_sem_proj[1].weight.sum() * 0.0
            return zero
        if entity_primitives is None or int(entity_primitives.shape[0]) == 0:
            return S_ent.float().sum() * 0.0
        if (
            entity_primitive_weights is None
            or not torch.is_tensor(entity_primitive_weights)
            or int(entity_primitive_weights.numel()) != int(entity_primitives.shape[0])
            or int(entity_primitive_weights.numel()) == 0
        ):
            return S_ent.float().sum() * 0.0

        w = entity_primitive_weights.float().to(device=entity_primitives.device).view(-1)
        w = w / w.sum().clamp_min(1e-6)
        target_ent = (entity_primitives.float() * w.unsqueeze(-1)).sum(dim=0, keepdim=True).detach()

        pooled_ent = S_ent.mean(dim=0, keepdim=True)
        pred_ent = self.pc_ssc_ent_sem_proj(
            pooled_ent.to(
                device=self.pc_ssc_ent_sem_proj[1].weight.device,
                dtype=self.pc_ssc_ent_sem_proj[1].weight.dtype,
            )
        )
        target_ent = target_ent.to(device=pred_ent.device, dtype=pred_ent.dtype)
        return (1.0 - F.cosine_similarity(pred_ent.float(), target_ent.float(), dim=-1)).mean()

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_pc_mlp_adapter = model_args.pretrain_pc_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type
        # newly added ones
        pointcloud_tower = model_args.pointcloud_tower
        pointcloud_decoder = model_args.pointcloud_decoder
        prompt_encoder = model_args.prompt_encoder
        inst_prompt_encoder = model_args.inst_prompt_encoder
        self._set_dvc_config(
            dvc_enable=getattr(model_args, "dvc_enable", False),
            dvc_zero_perturbation=getattr(model_args, "dvc_zero_perturbation", True),
            dvc_phase_schedule=getattr(model_args, "dvc_phase_schedule", "step"),
            dvc_warmup_ratio=getattr(model_args, "dvc_warmup_ratio", 0.1),
            dvc_entity_ratio=getattr(model_args, "dvc_entity_ratio", 0.7),
            dvc_layout_cells_per_axis=getattr(model_args, "dvc_layout_cells_per_axis", 6),
            dvc_role_embed=getattr(model_args, "dvc_role_embed", True),
            dvc_use_layout_adapter=getattr(model_args, "dvc_use_layout_adapter", True),
            dvc_use_role_embed=getattr(model_args, "dvc_use_role_embed", getattr(model_args, "dvc_role_embed", True)),
            dvc_use_layout_mix_gate=getattr(model_args, "dvc_use_layout_mix_gate", True),
            dvc_use_tail_role_heads=getattr(model_args, "dvc_use_tail_role_heads", True),
            dvc_use_slotwise_layout_mix=getattr(model_args, "dvc_use_slotwise_layout_mix", True),
            dvc_warmup_lr_scale=getattr(model_args, "dvc_warmup_lr_scale", 0.1),
            dvc_joint_lr_scale=getattr(model_args, "dvc_joint_lr_scale", 1.0),
            dvc_role_embed_zero_init=getattr(model_args, "dvc_role_embed_zero_init", True),
            dvc_variant=getattr(model_args, "dvc_variant", "legacy"),
            dvc_second_token_semantics=getattr(model_args, "dvc_second_token_semantics", "frame_anchor"),
            tsp_state_ent_slots=getattr(model_args, "tsp_state_ent_slots", getattr(model_args, "tsp_state_obj_slots", 4)),
            tsp_state_frame_slots=getattr(model_args, "tsp_state_frame_slots", getattr(model_args, "tsp_state_sup_slots", 4)),
            tsp_state_rel_slots=getattr(model_args, "tsp_state_rel_slots", 4),
            tsp_state_num_heads=getattr(model_args, "tsp_state_num_heads", 4),
            tsp_state_dropout=getattr(model_args, "tsp_state_dropout", 0.0),
            tsp_rel_seg_residual=getattr(model_args, "tsp_rel_seg_residual", getattr(model_args, "tsp_seg_query_residual", True)),
            tsp_rel_seg_alpha_init=getattr(model_args, "tsp_rel_seg_alpha_init", getattr(model_args, "tsp_seg_query_alpha_init", 0.0)),
            tsp_loss_sep=getattr(model_args, "tsp_loss_sep", 0.01),
            tsp_loss_div=getattr(model_args, "tsp_loss_div", 0.01),
            tsp_use_state_role_embed=getattr(model_args, "tsp_use_state_role_embed", True),
            tsp_use_ref_objects=getattr(model_args, "tsp_use_ref_objects", True),
            tsp_use_vertical_anchors=getattr(model_args, "tsp_use_vertical_anchors", True),
            tsp_ref_raw_tokens=getattr(model_args, "tsp_ref_raw_tokens", 8),
            tsp_struct_raw_tokens=getattr(model_args, "tsp_struct_raw_tokens", 8),
            tsp_debug_diversity=getattr(model_args, "tsp_debug_diversity", False),
            tsp_debug_diversity_print_once=getattr(model_args, "tsp_debug_diversity_print_once", True),
            tsp_use_slot_identity=getattr(model_args, "tsp_use_slot_identity", True),
            tsp_use_slot_competition=getattr(model_args, "tsp_use_slot_competition", True),
            tsp_rel_use_raw_evidence=getattr(model_args, "tsp_rel_use_raw_evidence", True),
            tsp_loss_attn_overlap=getattr(model_args, "tsp_loss_attn_overlap", 0.05),
            tsp_slot_identity_scale=getattr(model_args, "tsp_slot_identity_scale", 1.0),
            tsp_state_update_residual=getattr(model_args, "tsp_state_update_residual", True),
            tsp_state_update_alpha_init=getattr(model_args, "tsp_state_update_alpha_init", 1.0),
            tsp_slot_attention_temperature=getattr(model_args, "tsp_slot_attention_temperature", 1.0),
            tsp_rel_ref_raw_slots_cap=getattr(model_args, "tsp_rel_ref_raw_slots_cap", 8),
            tsp_rel_str_raw_slots_cap=getattr(model_args, "tsp_rel_str_raw_slots_cap", 8),
            tsp_rel_raw_bias=getattr(model_args, "tsp_rel_raw_bias", 2.0),
            tsp_rel_state_bias=getattr(model_args, "tsp_rel_state_bias", 0.5),
            tsp_loss_rel_raw_share=getattr(model_args, "tsp_loss_rel_raw_share", 0.05),
            tsp_rel_raw_share_target=getattr(model_args, "tsp_rel_raw_share_target", 0.65),
            tsp_rel_refine_enable=getattr(model_args, "tsp_rel_refine_enable", True),
            tsp_rel_refine_alpha_init=getattr(model_args, "tsp_rel_refine_alpha_init", 0.1),
            tsp_rel_refine_use_residual=getattr(model_args, "tsp_rel_refine_use_residual", True),
            tsp_rel_norm_control=getattr(model_args, "tsp_rel_norm_control", True),
            tsp_rel_norm_target_ratio=getattr(model_args, "tsp_rel_norm_target_ratio", 1.0),
            tsp_rel_norm_eps=getattr(model_args, "tsp_rel_norm_eps", 1e-6),
            tsp_sup_alpha_init=getattr(model_args, "tsp_sup_alpha_init", 0.5),
            tsp_sup_update_clamp=getattr(model_args, "tsp_sup_update_clamp", True),
            tsp_sup_update_max_scale=getattr(model_args, "tsp_sup_update_max_scale", 1.0),
            ssc_state_ent_slots=getattr(model_args, "ssc_state_ent_slots", 8),
            ssc_state_frame_slots=getattr(model_args, "ssc_state_frame_slots", 8),
            ssc_state_rel_slots=getattr(model_args, "ssc_state_rel_slots", 8),
            ssc_scene_summary_token_count=getattr(model_args, "ssc_scene_summary_token_count", 1),
            ssc_entity_candidate_topk=getattr(model_args, "ssc_entity_candidate_topk", 128),
            ssc_detail_tokens=getattr(model_args, "ssc_detail_tokens", -1),
            ssc_seg_alpha_init=getattr(model_args, "ssc_seg_alpha_init", 0.0),
            ssc_frame_cov_loss_weight=getattr(model_args, "ssc_frame_cov_loss_weight", 0.05),
            ssc_rel_boundary_loss_weight=getattr(model_args, "ssc_rel_boundary_loss_weight", 0.05),
            ssc_rel_region_loss_weight=getattr(model_args, "ssc_rel_region_loss_weight", 0.05),
            ssc_ent_sem_loss_weight=getattr(model_args, "ssc_ent_sem_loss_weight", 0.05),
            ssc_detail_dropout_rate=getattr(model_args, "ssc_detail_dropout_rate", 0.2),
            ssc_ablation_mode=getattr(model_args, "ssc_ablation_mode", "none"),
            ssc_role_analysis_variant=getattr(model_args, "ssc_role_analysis_variant", "none"),
        )


        self.config.mm_vision_tower = vision_tower
        if vision_tower is not None:
            if self.get_vision_tower() is None:
                vision_tower = build_vision_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.vision_tower = [vision_tower]
                else:
                    self.vision_tower = vision_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    vision_tower = self.vision_tower[0]
                else:
                    vision_tower = self.vision_tower
                vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = getattr(vision_tower, 'hidden_size', 1024)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        self.config.mm_pointcloud_tower = pointcloud_tower
        if pointcloud_tower is not None:
            if self.get_pointcloud_tower() is None:
                pointcloud_tower = build_pointcloud_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.pointcloud_tower = [pointcloud_tower]
                else:
                    self.pointcloud_tower = pointcloud_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    pointcloud_tower = self.pointcloud_tower[0]
                else:
                    pointcloud_tower = self.pointcloud_tower
                pointcloud_tower.load_model()

        self.config.use_pc_proj = True
        self.config.pc_hidden_size = pointcloud_tower.hidden_size
        self.config.pc_feature_dim = pointcloud_tower.feature_dim
        seg_hidden_size = getattr(self.config, "tsp_seg_hidden_size", None)
        if hasattr(pointcloud_tower, "seg_query_dim"):
            try:
                seg_hidden_size = int(pointcloud_tower.seg_query_dim)
            except Exception:
                pass
        self.config.tsp_seg_hidden_size = seg_hidden_size
        self.config.ssc_seg_hidden_size = seg_hidden_size

        if not bool(getattr(self.config, "tsp_enable", False)):
            self._maybe_initialize_pc_dvc_modules(self.config.pc_hidden_size)
        self._maybe_initialize_pc_tsp_modules(self.config.pc_hidden_size)
        self._maybe_initialize_pc_ssc_modules(self.config.pc_hidden_size)
        if bool(getattr(self.config, "tsp_enable", False)):
            dvc_module_names = []
            tsp_module_names = [
                "pc_state_query_ent",
                "pc_state_query_frame",
                "pc_state_query_rel",
                "pc_state_ent_block",
                "pc_state_frame_block",
                "pc_state_rel_block",
                "pc_rel_to_seg",
                "pc_rel_seg_alpha",
                "pc_role_embed",
            ]
            ssc_module_names = []
        elif str(getattr(self.config, "dvc_variant", "legacy")) == "ssc3d":
            dvc_module_names = []
            tsp_module_names = []
            ssc_module_names = [
                "pc_ssc_entity_primitive_head",
                "pc_ssc_frame_object_primitive_head",
                "pc_ssc_state_query_ent",
                "pc_ssc_state_query_frame",
                "pc_ssc_state_query_rel",
                "pc_ssc_state_query_generic",
                "pc_ssc_state_ent_block",
                "pc_ssc_state_frame_block",
                "pc_ssc_state_rel_block",
                "pc_ssc_state_generic_block",
                "pc_ssc_token_role_embed",
                "pc_ssc_generic_token_embed",
                "pc_ssc_seg_reader",
                "pc_ssc_seg_alpha",
                "pc_ssc_scene_proj",
                "pc_ssc_ent_sem_proj",
                "pc_ssc_rel_boundary_head",
                "pc_ssc_rel_region_head",
            ]
        else:
            dvc_module_names = [
                "pc_layout_adapter",
                "pc_entity_tail_role_head",
                "pc_layout_role_head",
                "pc_role_embed",
                "pc_layout_mix_gate",
                "pc_layout_slot_gate",
            ]
            tsp_module_names = [
                "pc_state_query_ent",
                "pc_state_query_frame",
                "pc_state_query_rel",
                "pc_state_ent_block",
                "pc_state_frame_block",
                "pc_state_rel_block",
                "pc_rel_to_seg",
                "pc_rel_seg_alpha",
                "pc_role_embed",
            ]
            ssc_module_names = []
        loaded_dvc_modules = []

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                prefix = keyword + '.'
                return {k.split(prefix, 1)[1]: v for k, v in weights.items() if prefix in k}

            def load_module_if_present(module, keyword):
                if module is None:
                    return
                module_state = get_w(mm_projector_weights, keyword)
                if len(module_state) > 0:
                    module.load_state_dict(module_state, strict=False)
                    loaded_dvc_modules.append(keyword)

            load_module_if_present(self.mm_projector, 'mm_projector')
            load_module_if_present(getattr(self.get_pointcloud_tower(), "alignment_proj", None), 'alignment_proj')
            load_module_if_present(getattr(self.get_pointcloud_tower(), "hidden_seg_fc", None), 'hidden_seg_fc')
            load_module_if_present(getattr(self, "pc_layout_adapter", None), 'pc_layout_adapter')
            load_module_if_present(getattr(self, "pc_entity_tail_role_head", None), 'pc_entity_tail_role_head')
            load_module_if_present(getattr(self, "pc_layout_role_head", None), 'pc_layout_role_head')
            load_module_if_present(getattr(self, "pc_layout_slot_gate", None), 'pc_layout_slot_gate')
            load_module_if_present(getattr(self, "lm_head_seg", None), 'lm_head_seg')
            load_module_if_present(getattr(self, "pc_state_ent_block", None), 'pc_state_ent_block')
            load_module_if_present(getattr(self, "pc_state_frame_block", None), 'pc_state_frame_block')
            load_module_if_present(getattr(self, "pc_state_rel_block", None), 'pc_state_rel_block')
            load_module_if_present(getattr(self, "pc_rel_to_seg", None), 'pc_rel_to_seg')
            load_module_if_present(getattr(self, "pc_ssc_entity_primitive_head", None), 'pc_ssc_entity_primitive_head')
            load_module_if_present(getattr(self, "pc_ssc_frame_object_primitive_head", None), 'pc_ssc_frame_object_primitive_head')
            load_module_if_present(getattr(self, "pc_ssc_state_ent_block", None), 'pc_ssc_state_ent_block')
            load_module_if_present(getattr(self, "pc_ssc_state_frame_block", None), 'pc_ssc_state_frame_block')
            load_module_if_present(getattr(self, "pc_ssc_state_rel_block", None), 'pc_ssc_state_rel_block')
            load_module_if_present(getattr(self, "pc_ssc_state_generic_block", None), 'pc_ssc_state_generic_block')
            load_module_if_present(getattr(self, "pc_ssc_seg_reader", None), 'pc_ssc_seg_reader')
            load_module_if_present(getattr(self, "pc_ssc_scene_proj", None), 'pc_ssc_scene_proj')
            load_module_if_present(getattr(self, "pc_ssc_ent_sem_proj", None), 'pc_ssc_ent_sem_proj')
            load_module_if_present(getattr(self, "pc_ssc_rel_boundary_head", None), 'pc_ssc_rel_boundary_head')
            load_module_if_present(getattr(self, "pc_ssc_rel_region_head", None), 'pc_ssc_rel_region_head')

            pc_role_embed = next(
                (v for k, v in mm_projector_weights.items() if k.endswith('pc_role_embed')),
                None,
            )
            if pc_role_embed is not None and hasattr(self, "pc_role_embed"):
                self.pc_role_embed.data.copy_(pc_role_embed)
                loaded_dvc_modules.append("pc_role_embed")

            pc_layout_mix_gate = next(
                (v for k, v in mm_projector_weights.items() if k.endswith('pc_layout_mix_gate')),
                None,
            )
            if pc_layout_mix_gate is not None and hasattr(self, "pc_layout_mix_gate"):
                self.pc_layout_mix_gate.data.copy_(pc_layout_mix_gate)
                loaded_dvc_modules.append("pc_layout_mix_gate")

            pc_state_query_ent = next(
                (
                    v for k, v in mm_projector_weights.items()
                    if k.endswith('pc_state_query_ent') or k.endswith('pc_state_query_obj')
                ),
                None,
            )
            if pc_state_query_ent is not None and hasattr(self, "pc_state_query_ent"):
                self.pc_state_query_ent.data.copy_(pc_state_query_ent)
                loaded_dvc_modules.append("pc_state_query_ent")
            pc_state_query_frame = next(
                (
                    v for k, v in mm_projector_weights.items()
                    if k.endswith('pc_state_query_frame') or k.endswith('pc_state_query_sup')
                ),
                None,
            )
            if pc_state_query_frame is not None and hasattr(self, "pc_state_query_frame"):
                self.pc_state_query_frame.data.copy_(pc_state_query_frame)
                loaded_dvc_modules.append("pc_state_query_frame")
            pc_state_query_rel = next((v for k, v in mm_projector_weights.items() if k.endswith('pc_state_query_rel')), None)
            if pc_state_query_rel is not None and hasattr(self, "pc_state_query_rel"):
                self.pc_state_query_rel.data.copy_(pc_state_query_rel)
                loaded_dvc_modules.append("pc_state_query_rel")
            def _copy_scalar_param_compat(dst_param, src_tensor):
                if src_tensor is None:
                    return False
                src = src_tensor
                if not isinstance(src, torch.Tensor):
                    return False
                if src.ndim == 0 and dst_param.ndim == 1 and int(dst_param.numel()) == 1:
                    src = src.unsqueeze(0)
                if tuple(src.shape) != tuple(dst_param.shape):
                    return False
                dst_param.data.copy_(src.to(device=dst_param.device, dtype=dst_param.dtype))
                return True
            pc_rel_seg_alpha = next(
                (
                    v for k, v in mm_projector_weights.items()
                    if k.endswith('pc_rel_seg_alpha') or k.endswith('pc_state_seg_alpha')
                ),
                None,
            )
            if pc_rel_seg_alpha is not None and hasattr(self, "pc_rel_seg_alpha"):
                if _copy_scalar_param_compat(self.pc_rel_seg_alpha, pc_rel_seg_alpha):
                    loaded_dvc_modules.append("pc_rel_seg_alpha")

            pc_ssc_token_role_embed = next(
                (v for k, v in mm_projector_weights.items() if k.endswith('pc_ssc_token_role_embed')),
                None,
            )
            if (
                pc_ssc_token_role_embed is not None
                and hasattr(self, "pc_ssc_token_role_embed")
                and tuple(pc_ssc_token_role_embed.shape) == tuple(self.pc_ssc_token_role_embed.shape)
            ):
                self.pc_ssc_token_role_embed.data.copy_(
                    pc_ssc_token_role_embed.to(
                        device=self.pc_ssc_token_role_embed.device,
                        dtype=self.pc_ssc_token_role_embed.dtype,
                    )
                )
                loaded_dvc_modules.append("pc_ssc_token_role_embed")

            pc_ssc_generic_token_embed = next(
                (v for k, v in mm_projector_weights.items() if k.endswith('pc_ssc_generic_token_embed')),
                None,
            )
            if (
                pc_ssc_generic_token_embed is not None
                and hasattr(self, "pc_ssc_generic_token_embed")
                and tuple(pc_ssc_generic_token_embed.shape) == tuple(self.pc_ssc_generic_token_embed.shape)
            ):
                self.pc_ssc_generic_token_embed.data.copy_(
                    pc_ssc_generic_token_embed.to(
                        device=self.pc_ssc_generic_token_embed.device,
                        dtype=self.pc_ssc_generic_token_embed.dtype,
                    )
                )
                loaded_dvc_modules.append("pc_ssc_generic_token_embed")

            for param_name in [
                "pc_ssc_state_query_ent",
                "pc_ssc_state_query_frame",
                "pc_ssc_state_query_rel",
                "pc_ssc_state_query_generic",
            ]:
                src_param = next((v for k, v in mm_projector_weights.items() if k.endswith(param_name)), None)
                dst_param = getattr(self, param_name, None)
                if src_param is not None and isinstance(dst_param, nn.Parameter) and tuple(src_param.shape) == tuple(dst_param.shape):
                    dst_param.data.copy_(src_param.to(device=dst_param.device, dtype=dst_param.dtype))
                    loaded_dvc_modules.append(param_name)

            pc_ssc_seg_alpha = next((v for k, v in mm_projector_weights.items() if k.endswith('pc_ssc_seg_alpha')), None)
            if pc_ssc_seg_alpha is not None and hasattr(self, "pc_ssc_seg_alpha"):
                if _copy_scalar_param_compat(self.pc_ssc_seg_alpha, pc_ssc_seg_alpha):
                    loaded_dvc_modules.append("pc_ssc_seg_alpha")

        if self.config.dvc_enable:
            loaded_dvc_modules = sorted(set(loaded_dvc_modules))
            target_modules = list(dvc_module_names)
            if bool(getattr(self.config, "tsp_enable", False)):
                target_modules = target_modules + tsp_module_names
            if str(getattr(self.config, "dvc_variant", "legacy")) == "ssc3d":
                target_modules = target_modules + ssc_module_names
            missing_dvc_modules = [name for name in target_modules if name not in loaded_dvc_modules]
            newly_initialized_modules = [name for name in missing_dvc_modules if hasattr(self, name)]
            print(
                "DVC module load summary: "
                f"loaded={loaded_dvc_modules}, "
                f"missing={missing_dvc_modules}, "
                f"newly_initialized={newly_initialized_modules}"
            )

        # build visual sampler
        self.config.mm_inst_prompt_encoder = inst_prompt_encoder
        if inst_prompt_encoder is not None:
            if self.get_inst_prompt_encoder() is None:
                if self.config.mm_inst_prompt_encoder not in ["shared_projector"]:
                #     self.inst_prompt_encoder = pointcloud_tower.alignment_proj
                # else:
                    self.inst_prompt_encoder = build_inst_prompt_encoder(self.config)


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_pointcloud_tower(self):
        return self.get_model().get_pointcloud_tower()

    def get_pointcloud_decoder(self):
        return self.get_model().get_pointcloud_decoder()

    def get_mask_decoder(self):
        return self.get_model().get_mask_decoder()

    def get_hidden_seg_fc(self):
        return self.get_model().get_hidden_seg_fc()

    def get_seg_criteria(self):
        return self.get_model().get_seg_criteria()
    
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def _normalize_second_token_semantics(self, semantics, dvc_variant):
        semantics = str(semantics)
        if str(dvc_variant) not in {"tsp3d", "ssc3d"}:
            return semantics
        if semantics in {"generic_layout", "relation_anchor", "hybrid_ra", "gta_lite"}:
            return "frame_anchor"
        return semantics

    def encode_pointclouds(
        self, coord, grid_coord, offset, feat, p2v_map, v2p_map, spatial_shape, superpoint_mask, prompt_mask,
        entity_selector="learned", selector_seed=0, selector_scene_id=None, ssc_state_override=None,
        ):
        baseline_entity_features_full, frame_anchor_features, prompt_features, superpoint_features, mask_input_dict = self.get_model().get_pointcloud_tower()(
            coord, grid_coord, offset, feat, p2v_map, v2p_map, spatial_shape, superpoint_mask, prompt_mask
        )
        self.get_model()._last_ssc_state_vis = None

        entity_token_counts = [int(feat.shape[0]) for feat in baseline_entity_features_full]
        frame_token_counts = [int(feat.shape[0]) for feat in frame_anchor_features]
        frame_valid_counts = [int(v) for v in mask_input_dict.get("frame_valid_counts", frame_token_counts)]
        if len(frame_valid_counts) < len(frame_token_counts):
            frame_valid_counts = frame_valid_counts + frame_token_counts[len(frame_valid_counts):]
        frame_valid_counts = frame_valid_counts[:len(frame_token_counts)]
        # Legacy path still consumes these names.
        layout_token_counts = frame_token_counts
        gate_scale = float(getattr(self.config, "current_dvc_gate_scale", 1.0))
        final_token_count_match = True
        used_ctx_second_counts = []
        used_ra_second_counts = []
        second_path_used_counts = []
        second_path_fill_ratio = []
        ra_slot_ratio = []

        dvc_variant = str(getattr(self.config, "dvc_variant", "legacy"))
        if dvc_variant == "tsp3d":
            configured_second_semantics = "frame_anchor"
        else:
            configured_second_semantics = self._normalize_second_token_semantics(
                getattr(self.config, "dvc_second_token_semantics", "frame_anchor"),
                dvc_variant,
            )
        if getattr(self.config, "dvc_enable", False) and dvc_variant == "tsp3d":
            if getattr(self.config, "tsp_seg_hidden_size", None) is None:
                pointcloud_tower = self.get_pointcloud_tower()
                if pointcloud_tower is not None and hasattr(pointcloud_tower, "seg_query_dim"):
                    try:
                        self.config.tsp_seg_hidden_size = int(pointcloud_tower.seg_query_dim)
                    except Exception:
                        pass
            self.get_model()._maybe_initialize_pc_tsp_modules(self.config.pc_hidden_size)

            pointcloud_tokens = []
            tsp_state_ent = []
            tsp_state_frame = []
            tsp_state_rel = []
            tsp_ent_state_norm = []
            tsp_frame_state_norm = []
            tsp_rel_state_norm = []

            for baseline_pc, frame_pc, scene_meta in zip(
                baseline_entity_features_full,
                frame_anchor_features,
                mask_input_dict.get("dvc_meta", []),
            ):
                keep_count = min(int(scene_meta.get("num_entity_keep", baseline_pc.shape[0])), int(baseline_pc.shape[0]))
                frame_slots_target = max(
                    int(scene_meta.get("num_frame_slots", int(baseline_pc.shape[0] - keep_count))),
                    0,
                )
                frame_slots_target = min(frame_slots_target, int(baseline_pc.shape[0] - keep_count))
                valid_frame_count = min(
                    int(scene_meta.get("num_frame_valid", frame_pc.shape[0])),
                    int(frame_pc.shape[0]),
                    int(frame_slots_target),
                )
                valid_frame_count = max(int(valid_frame_count), 0)

                entity_head_pc = baseline_pc[:keep_count]
                frame_evidence_valid = frame_pc[:valid_frame_count]
                frame_tail_pc_for_tokens = frame_evidence_valid
                if valid_frame_count < frame_slots_target:
                    pad_count = int(frame_slots_target - valid_frame_count)
                    frame_tail_pc_for_tokens = torch.cat(
                        [frame_tail_pc_for_tokens, baseline_pc.new_zeros((pad_count, baseline_pc.shape[-1]))],
                        dim=0,
                    )

                entity_head_mm = self.get_model().mm_projector(entity_head_pc)
                frame_tail_mm = self.get_model().mm_projector(frame_tail_pc_for_tokens)
                if bool(getattr(self.config, "dvc_use_role_embed", True)) and hasattr(self.get_model(), "pc_role_embed"):
                    role_ent = self.get_model().pc_role_embed[0].to(
                        device=entity_head_mm.device,
                        dtype=entity_head_mm.dtype,
                    )
                    role_frame = self.get_model().pc_role_embed[1].to(
                        device=frame_tail_mm.device,
                        dtype=frame_tail_mm.dtype,
                    )
                    entity_head_mm = entity_head_mm + role_ent
                    frame_tail_mm = frame_tail_mm + role_frame

                cur_tokens = torch.cat([entity_head_mm, frame_tail_mm], dim=0)
                final_token_count_match = final_token_count_match and (int(cur_tokens.shape[0]) == int(baseline_pc.shape[0]))
                if int(cur_tokens.shape[0]) != int(baseline_pc.shape[0]):
                    raise RuntimeError(
                        f"TSP3D final token count mismatch: got {int(cur_tokens.shape[0])}, expected {int(baseline_pc.shape[0])}."
                    )
                pointcloud_tokens.append(cur_tokens)

                zero_token = baseline_pc.new_zeros((1, baseline_pc.shape[-1]))
                entity_evidence = entity_head_pc if int(entity_head_pc.shape[0]) > 0 else zero_token
                if int(frame_evidence_valid.shape[0]) == 0:
                    frame_evidence_valid = zero_token
                q_ent = self.get_model().pc_state_query_ent.to(device=baseline_pc.device, dtype=baseline_pc.dtype)
                s_ent, _ = self.get_model().pc_state_ent_block(q_ent, entity_evidence, return_attn=True)
                q_frame = self.get_model().pc_state_query_frame.to(device=baseline_pc.device, dtype=baseline_pc.dtype)
                s_frame, _ = self.get_model().pc_state_frame_block(q_frame, frame_evidence_valid, return_attn=True)
                rel_source = torch.cat([s_ent, s_frame], dim=0)
                q_rel = self.get_model().pc_state_query_rel.to(device=baseline_pc.device, dtype=baseline_pc.dtype)
                s_rel, _ = self.get_model().pc_state_rel_block(q_rel, rel_source, return_attn=True)

                tsp_state_ent.append(s_ent)
                tsp_state_frame.append(s_frame)
                tsp_state_rel.append(s_rel)
                tsp_ent_state_norm.append(float(s_ent.norm(dim=-1).mean().item()) if int(s_ent.shape[0]) > 0 else 0.0)
                tsp_frame_state_norm.append(float(s_frame.norm(dim=-1).mean().item()) if int(s_frame.shape[0]) > 0 else 0.0)
                tsp_rel_state_norm.append(float(s_rel.norm(dim=-1).mean().item()) if int(s_rel.shape[0]) > 0 else 0.0)

            boundary_selected_counts = [int(v) for v in mask_input_dict.get("boundary_selected_counts", [0] * len(frame_token_counts))]
            region_selected_counts = [int(v) for v in mask_input_dict.get("region_selected_counts", [0] * len(frame_token_counts))]
            if len(boundary_selected_counts) < len(frame_token_counts):
                boundary_selected_counts = boundary_selected_counts + [0] * (len(frame_token_counts) - len(boundary_selected_counts))
            if len(region_selected_counts) < len(frame_token_counts):
                region_selected_counts = region_selected_counts + [0] * (len(frame_token_counts) - len(region_selected_counts))
            boundary_selected_counts = boundary_selected_counts[:len(frame_token_counts)]
            region_selected_counts = region_selected_counts[:len(frame_token_counts)]

            prompt_tokens = [self.get_model().mm_projector(feat) for feat in prompt_features]
            mask_input_dict["entity_token_counts"] = entity_token_counts
            mask_input_dict["frame_token_counts"] = frame_token_counts
            mask_input_dict["frame_valid_counts"] = frame_valid_counts
            mask_input_dict["boundary_selected_counts"] = boundary_selected_counts
            mask_input_dict["region_selected_counts"] = region_selected_counts
            mask_input_dict["tsp_state_bank"] = {
                "S_ent": tsp_state_ent,
                "S_frame": tsp_state_frame,
                "S_rel": tsp_state_rel,
            }
            tsp_debug = {
                "ent_state_norm_mean": float(sum(tsp_ent_state_norm) / float(max(len(tsp_ent_state_norm), 1))),
                "frame_state_norm_mean": float(sum(tsp_frame_state_norm) / float(max(len(tsp_frame_state_norm), 1))),
                "rel_state_norm_mean": float(sum(tsp_rel_state_norm) / float(max(len(tsp_rel_state_norm), 1))),
                "frame_token_counts": frame_token_counts,
                "frame_valid_counts": frame_valid_counts,
                "boundary_selected_counts": boundary_selected_counts,
                "region_selected_counts": region_selected_counts,
                "final_token_count_match": bool(final_token_count_match),
                "tsp_rel_seg_alpha": float(self.get_model().pc_rel_seg_alpha.detach().item()),
            }
            mask_input_dict["tsp_debug"] = tsp_debug
            self.get_model()._last_tsp_debug = tsp_debug

            final_token_counts = [int(tok.shape[0]) for tok in pointcloud_tokens]
            mask_input_dict["dvc_debug"] = {
                "dvc_variant": dvc_variant,
                "dvc_second_token_semantics": configured_second_semantics,
                "entity_token_counts": entity_token_counts,
                "frame_token_counts": frame_token_counts,
                "frame_valid_counts": frame_valid_counts,
                "boundary_selected_counts": boundary_selected_counts,
                "region_selected_counts": region_selected_counts,
                "ent_state_norm_mean": tsp_debug["ent_state_norm_mean"],
                "frame_state_norm_mean": tsp_debug["frame_state_norm_mean"],
                "rel_state_norm_mean": tsp_debug["rel_state_norm_mean"],
                "final_token_counts": final_token_counts,
                "final_token_count_match": bool(final_token_count_match),
            }
            self.get_model()._last_dvc_debug = mask_input_dict["dvc_debug"]
            return pointcloud_tokens, prompt_tokens, superpoint_features, mask_input_dict

        elif getattr(self.config, "dvc_enable", False) and dvc_variant == "ssc3d":
            if getattr(self.config, "ssc_seg_hidden_size", None) is None:
                pointcloud_tower = self.get_pointcloud_tower()
                if pointcloud_tower is not None and hasattr(pointcloud_tower, "seg_query_dim"):
                    try:
                        self.config.ssc_seg_hidden_size = int(pointcloud_tower.seg_query_dim)
                    except Exception:
                        pass
            self.get_model()._maybe_initialize_pc_ssc_modules(self.config.pc_hidden_size)

            num_pc_tokens = int(getattr(self.config, "num_pc_tokens", 0))
            if num_pc_tokens <= 0 and len(baseline_entity_features_full) > 0:
                num_pc_tokens = int(baseline_entity_features_full[0].shape[0])
            ssc_layout = self.get_model()._resolve_ssc_ablation_layout(num_pc_tokens=num_pc_tokens)
            ent_slots = int(ssc_layout["original_ent_slots"])
            frame_slots = int(ssc_layout["original_frame_slots"])
            rel_slots = int(ssc_layout["original_rel_slots"])
            use_ent = bool(ssc_layout["use_ent"])
            use_frame = bool(ssc_layout["use_frame"])
            use_rel = bool(ssc_layout["use_rel"])
            use_scene = bool(ssc_layout["use_scene"])
            scene_summary_token_count = int(ssc_layout["effective_scene_summary_token_count"])
            detail_tokens = int(ssc_layout["effective_detail_tokens"])
            if int(ssc_layout["final_token_count"]) != int(num_pc_tokens):
                raise RuntimeError(
                    "SSC3D final token budget mismatch: "
                    f"detail_tokens={detail_tokens}, "
                    f"effective_state_total={int(ssc_layout['effective_state_total'])}, "
                    f"num_pc_tokens={num_pc_tokens}."
                )

            ssc_role_analysis = self.get_model()._resolve_ssc_role_analysis_settings(num_pc_tokens=num_pc_tokens)
            pointcloud_tokens = []
            ssc_state_ent = []
            ssc_state_frame = []
            ssc_state_rel = []
            ssc_state_scene = []
            ssc_state_generic = []
            ssc_ent_state_norm = []
            ssc_frame_state_norm = []
            ssc_rel_state_norm = []
            ssc_generic_state_norm = []
            ssc_frame_cov_losses = []
            ssc_rel_boundary_losses = []
            ssc_rel_region_losses = []
            ssc_ent_sem_losses = []
            ssc_used_frame_evidence_counts = []
            ssc_frame_object_token_counts = []
            final_token_count_match = True
            capture_ssc_state_vis = bool(getattr(self.get_model(), "_capture_ssc_state_vis", False))
            ssc_state_vis_records = [] if capture_ssc_state_vis else None
            ssc_state_override_records = []
            if ssc_state_override is not None and not isinstance(ssc_state_override, dict):
                raise TypeError("ssc_state_override must be a mapping of role names to donor state tensors")
            supported_override_roles = {"S_ent", "S_frame", "S_rel"}
            if isinstance(ssc_state_override, dict):
                unknown_override_roles = set(ssc_state_override) - supported_override_roles
                if unknown_override_roles:
                    raise ValueError(
                        "unsupported ssc_state_override roles: "
                        + ", ".join(sorted(str(role) for role in unknown_override_roles))
                    )

            scene_meta_list = list(mask_input_dict.get("dvc_meta", []))
            if len(scene_meta_list) < len(baseline_entity_features_full):
                scene_meta_list = scene_meta_list + [{} for _ in range(len(baseline_entity_features_full) - len(scene_meta_list))]
            sp_xyz_list = mask_input_dict.get("sp_xyz", [])
            frame_group_ids_list = mask_input_dict.get("frame_group_ids", [])
            frame_anchor_coord_list = mask_input_dict.get("frame_anchor_coord", [])
            frame_primitive_budget = [
                int(v) for v in mask_input_dict.get("ssc_frame_primitive_budget", [0] * len(frame_token_counts))
            ]
            if len(frame_primitive_budget) < len(frame_token_counts):
                frame_primitive_budget = frame_primitive_budget + [0] * (len(frame_token_counts) - len(frame_primitive_budget))
            frame_primitive_budget = frame_primitive_budget[:len(frame_token_counts)]
            ssc_detail_tokens_resolved = [
                int(v) for v in mask_input_dict.get("ssc_detail_tokens_resolved", [int(detail_tokens)] * len(frame_token_counts))
            ]
            if len(ssc_detail_tokens_resolved) < len(frame_token_counts):
                ssc_detail_tokens_resolved = ssc_detail_tokens_resolved + [int(detail_tokens)] * (len(frame_token_counts) - len(ssc_detail_tokens_resolved))
            ssc_detail_tokens_resolved = ssc_detail_tokens_resolved[:len(frame_token_counts)]
            ssc_baseline_entity_request_tokens = [
                int(v) for v in mask_input_dict.get("ssc_baseline_entity_request_tokens", [0] * len(frame_token_counts))
            ]
            if len(ssc_baseline_entity_request_tokens) < len(frame_token_counts):
                ssc_baseline_entity_request_tokens = ssc_baseline_entity_request_tokens + [0] * (len(frame_token_counts) - len(ssc_baseline_entity_request_tokens))
            ssc_baseline_entity_request_tokens = ssc_baseline_entity_request_tokens[:len(frame_token_counts)]

            entity_candidate_topk = max(int(getattr(self.config, "ssc_entity_candidate_topk", 128)), 0)
            generic_state_slots = int(ssc_role_analysis["generic_state_slots"])

            def _select_topk_primitives(source_features, source_xyz, primitive_head, topk_budget, fallback_zero_token, selector="learned", scene_key="", seed=0):
                if (
                    primitive_head is None
                    or source_features is None
                    or source_xyz is None
                    or int(topk_budget) <= 0
                    or int(source_features.shape[0]) <= 0
                ):
                    return (
                        fallback_zero_token,
                        source_xyz.new_zeros((0, source_xyz.shape[-1])),
                        None,
                        0,
                        source_xyz.new_zeros((0,), dtype=torch.long),
                        source_features.new_zeros((0,)) if source_features is not None else None,
                    )
                scores = primitive_head(
                    source_features.to(dtype=primitive_head[1].weight.dtype)
                ).squeeze(-1)
                k_selected = min(int(topk_budget), int(source_features.shape[0]))
                if k_selected <= 0:
                    return (
                        fallback_zero_token,
                        source_xyz.new_zeros((0, source_xyz.shape[-1])),
                        None,
                        0,
                        source_xyz.new_zeros((0,), dtype=torch.long),
                        scores.float(),
                    )
                selector = str(selector or "learned").lower()
                if selector == "learned":
                    topk_idx = scores.float().topk(k_selected, sorted=True).indices
                elif selector == "bottom":
                    topk_idx = torch.argsort(scores.float(), stable=True)[:k_selected]
                elif selector == "random":
                    key = f"{int(seed)}:{scene_key}".encode("utf-8")
                    stable_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)
                    generator = torch.Generator(device=source_features.device)
                    generator.manual_seed(stable_seed)
                    topk_idx = torch.randperm(int(source_features.shape[0]), generator=generator, device=source_features.device)[:k_selected]
                else:
                    raise ValueError(f"unknown entity selector: {selector}")
                topk_scores = scores.float()[topk_idx]
                primitives = source_features[topk_idx]
                primitive_xyz = source_xyz[topk_idx]
                primitive_weights = torch.softmax(topk_scores, dim=0) if int(topk_scores.numel()) > 0 else None
                return primitives, primitive_xyz, primitive_weights, k_selected, topk_idx, scores.float()

            def _tensor_to_cpu_float(value):
                if value is None:
                    return None
                return value.detach().to(torch.float32).cpu()

            def _tensor_to_cpu_long(value):
                if value is None:
                    return None
                return value.detach().to(torch.long).cpu()

            def _weighted_state_to_evidence(slot_scores, slot_to_evidence):
                if slot_scores is None or slot_to_evidence is None:
                    return None
                if int(slot_scores.numel()) == 0 or int(slot_to_evidence.numel()) == 0:
                    return None
                weights = slot_scores.float().view(-1)
                denom = weights.sum().clamp_min(1e-6)
                return torch.matmul(weights.unsqueeze(0), slot_to_evidence.float()).squeeze(0) / denom

            profile_ssc_constructor = bool(getattr(self.get_model(), "_e7_profile_ssc_constructor", False))
            ssc_constructor_start = None
            ssc_constructor_end = None
            if profile_ssc_constructor:
                ssc_constructor_start = torch.cuda.Event(enable_timing=True)
                ssc_constructor_end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                ssc_constructor_start.record()

            for scene_idx, (baseline_pc, frame_pc, sp_feat) in enumerate(
                zip(baseline_entity_features_full, frame_anchor_features, superpoint_features)
            ):
                scene_meta = scene_meta_list[scene_idx]
                if scene_idx >= len(sp_xyz_list) or sp_xyz_list[scene_idx] is None:
                    raise RuntimeError(
                        "SSC3D requires valid sp_xyz for geometry-aware state construction, "
                        f"but scene_idx={scene_idx} has no sp_xyz."
                    )
                sp_xyz = sp_xyz_list[scene_idx]
                if int(sp_xyz.shape[0]) != int(sp_feat.shape[0]):
                    raise RuntimeError(
                        "SSC3D expects sp_xyz and superpoint_features to have the same length, "
                        f"but got sp_xyz={int(sp_xyz.shape[0])}, superpoint_features={int(sp_feat.shape[0])} "
                        f"for scene_idx={scene_idx}."
                    )

                pc_dim = int(baseline_pc.shape[-1]) if baseline_pc.ndim > 1 else int(getattr(self.config, "pc_hidden_size", 1024))
                zero_token = baseline_pc.new_zeros((1, pc_dim))
                zero_loss = baseline_pc.float().sum() * 0.0

                need_entity_primitives = bool(
                    use_ent
                    or use_rel
                    or ssc_role_analysis["entity_evidence_source"] == "shared_object_pool"
                    or ssc_role_analysis["frame_evidence_source"] == "shared_object_pool"
                    or ssc_role_analysis["relation_source"] == "primitive_evidence"
                )
                if need_entity_primitives:
                    (
                        entity_primitives,
                        entity_primitive_xyz,
                        entity_primitive_weights,
                        k_entity_prim,
                        entity_topk_idx,
                        entity_all_scores,
                    ) = _select_topk_primitives(
                        sp_feat,
                        sp_xyz,
                        getattr(self.get_model(), "pc_ssc_entity_primitive_head", None),
                        entity_candidate_topk,
                        zero_token,
                        selector=entity_selector,
                        scene_key=str(selector_scene_id or scene_meta.get("scene_id", scene_idx)),
                        seed=int(selector_seed),
                    )
                else:
                    entity_primitives = zero_token
                    entity_primitive_xyz = sp_xyz.new_zeros((0, sp_xyz.shape[-1]))
                    entity_primitive_weights = None
                    k_entity_prim = 0
                    entity_topk_idx = sp_xyz.new_zeros((0,), dtype=torch.long)
                    entity_all_scores = sp_feat.new_zeros((0,))

                if use_frame:
                    frame_valid = min(
                        int(scene_meta.get("num_frame_valid", int(frame_pc.shape[0]))),
                        int(frame_pc.shape[0]),
                    )
                    frame_valid = max(int(frame_valid), 0)
                    frame_primitives_valid = frame_pc[:frame_valid]
                    frame_anchor_coord_valid = (
                        frame_anchor_coord_list[scene_idx][:frame_valid]
                        if scene_idx < len(frame_anchor_coord_list)
                        else sp_xyz.new_zeros((0, sp_xyz.shape[-1]))
                    )
                    frame_group_ids = (
                        frame_group_ids_list[scene_idx]
                        if scene_idx < len(frame_group_ids_list)
                        else scene_meta.get("selected_group_ids", [])
                    )
                    frame_group_ids_valid = [int(v) for v in list(frame_group_ids)[:frame_valid]]
                    frame_group_kind = list(scene_meta.get("selected_group_kind", []))
                    frame_group_kind_valid = [int(v) for v in frame_group_kind[:frame_valid]]
                    frame_primitives = frame_primitives_valid if frame_valid > 0 else zero_token
                else:
                    frame_valid = 0
                    frame_anchor_coord_valid = sp_xyz.new_zeros((0, sp_xyz.shape[-1]))
                    frame_group_ids_valid = []
                    frame_group_kind_valid = []
                    frame_primitives = zero_token
                if ssc_role_analysis["frame_evidence_source"] == "frame_object_topk":
                    (
                        frame_object_primitives,
                        frame_object_primitive_xyz,
                        frame_object_primitive_weights,
                        k_frame_object_prim,
                        frame_object_topk_idx,
                        frame_object_all_scores,
                    ) = _select_topk_primitives(
                        sp_feat,
                        sp_xyz,
                        getattr(self.get_model(), "pc_ssc_frame_object_primitive_head", None),
                        entity_candidate_topk,
                        zero_token,
                    )
                else:
                    frame_object_primitives = zero_token
                    frame_object_primitive_xyz = sp_xyz.new_zeros((0, sp_xyz.shape[-1]))
                    frame_object_primitive_weights = None
                    k_frame_object_prim = 0
                    frame_object_topk_idx = sp_xyz.new_zeros((0,), dtype=torch.long)
                    frame_object_all_scores = sp_feat.new_zeros((0,))
                ssc_frame_object_token_counts.append(int(k_frame_object_prim))

                if bool(ssc_role_analysis["use_generic_states"]):
                    generic_evidence = sp_feat if int(sp_feat.shape[0]) > 0 else zero_token
                    q_generic = self.get_model().pc_ssc_state_query_generic.to(
                        device=generic_evidence.device, dtype=generic_evidence.dtype
                    )
                    S_generic, _ = self.get_model().pc_ssc_state_generic_block(
                        q_generic, generic_evidence, return_attn=True
                    )
                    S_ent = baseline_pc.new_zeros((0, pc_dim))
                    S_frame = baseline_pc.new_zeros((0, pc_dim))
                    S_rel = baseline_pc.new_zeros((0, pc_dim))
                    S_scene = baseline_pc.new_zeros((0, pc_dim))
                    attn_ent = None
                    attn_frame = None
                    attn_rel = None
                    entity_state_evidence = baseline_pc.new_zeros((0, pc_dim))
                    entity_state_weights = None
                    frame_state_evidence = baseline_pc.new_zeros((0, pc_dim))
                    frame_state_evidence_xyz = sp_xyz.new_zeros((0, sp_xyz.shape[-1]))
                    frame_state_group_ids_valid = []
                    frame_state_group_kind_valid = []
                    frame_state_evidence_count = 0
                else:
                    if ssc_role_analysis["entity_evidence_source"] == "shared_object_pool":
                        entity_state_evidence = entity_primitives
                        entity_state_weights = entity_primitive_weights
                    else:
                        entity_state_evidence = entity_primitives
                        entity_state_weights = entity_primitive_weights

                    if ssc_role_analysis["frame_evidence_source"] == "shared_object_pool":
                        frame_state_evidence = entity_primitives
                        frame_state_evidence_xyz = entity_primitive_xyz
                        frame_state_group_ids_valid = []
                        frame_state_group_kind_valid = []
                        frame_state_evidence_count = int(k_entity_prim)
                    elif ssc_role_analysis["frame_evidence_source"] == "frame_object_topk":
                        frame_state_evidence = frame_object_primitives
                        frame_state_evidence_xyz = frame_object_primitive_xyz
                        frame_state_group_ids_valid = []
                        frame_state_group_kind_valid = []
                        frame_state_evidence_count = int(k_frame_object_prim)
                    else:
                        frame_state_evidence = frame_primitives
                        frame_state_evidence_xyz = frame_anchor_coord_valid
                        frame_state_group_ids_valid = frame_group_ids_valid
                        frame_state_group_kind_valid = frame_group_kind_valid
                        frame_state_evidence_count = int(frame_valid)

                    if use_ent:
                        q_ent = self.get_model().pc_ssc_state_query_ent.to(
                            device=entity_state_evidence.device, dtype=entity_state_evidence.dtype
                        )
                        S_ent, attn_ent = self.get_model().pc_ssc_state_ent_block(
                            q_ent, entity_state_evidence, return_attn=True
                        )
                    else:
                        ent_placeholder_slots = max(ent_slots, 1)
                        S_ent = baseline_pc.new_zeros((ent_placeholder_slots, pc_dim))
                        attn_ent = None

                    if use_frame:
                        q_frame = self.get_model().pc_ssc_state_query_frame.to(
                            device=frame_state_evidence.device, dtype=frame_state_evidence.dtype
                        )
                        S_frame, attn_frame = self.get_model().pc_ssc_state_frame_block(
                            q_frame, frame_state_evidence, return_attn=True
                        )
                    else:
                        frame_placeholder_slots = max(frame_slots, 1)
                        S_frame = baseline_pc.new_zeros((frame_placeholder_slots, pc_dim))
                        attn_frame = None

                    if ssc_role_analysis["relation_source"] == "primitive_evidence":
                        rel_source = torch.cat([entity_primitives, frame_state_evidence], dim=0)
                    else:
                        rel_source = torch.cat([S_ent, S_frame], dim=0)
                    if use_rel:
                        q_rel = self.get_model().pc_ssc_state_query_rel.to(
                            device=rel_source.device, dtype=rel_source.dtype
                        )
                        S_rel, attn_rel = self.get_model().pc_ssc_state_rel_block(
                            q_rel, rel_source, return_attn=True
                        )
                    else:
                        rel_placeholder_slots = max(rel_slots, 1)
                        S_rel = baseline_pc.new_zeros((rel_placeholder_slots, pc_dim))
                        attn_rel = None

                    pooled_ent = S_ent.mean(dim=0, keepdim=True)
                    pooled_frame = S_frame.mean(dim=0, keepdim=True)
                    pooled_rel = S_rel.mean(dim=0, keepdim=True)
                    if use_scene:
                        pooled_scene = torch.cat([pooled_ent, pooled_frame, pooled_rel], dim=-1)
                        scene_proj = self.get_model().pc_ssc_scene_proj
                        scene_raw = scene_proj(
                            pooled_scene.to(device=scene_proj.weight.device, dtype=scene_proj.weight.dtype)
                        ).to(device=S_ent.device, dtype=S_ent.dtype)
                        S_scene = scene_raw.view(max(scene_summary_token_count, 1), pc_dim)
                    else:
                        S_scene = baseline_pc.new_zeros((0, pc_dim))
                    S_generic = baseline_pc.new_zeros((0, pc_dim))

                # Apply fixed-weight donor states before projection into visual tokens.
                # Attention tensors above intentionally remain the original construction path.
                override_roles = []
                if isinstance(ssc_state_override, dict):
                    for role_name, state_list, state_value in (
                        ("S_ent", ssc_state_ent, S_ent),
                        ("S_frame", ssc_state_frame, S_frame),
                        ("S_rel", ssc_state_rel, S_rel),
                    ):
                        if role_name not in ssc_state_override:
                            continue
                        donor = ssc_state_override[role_name]
                        if isinstance(donor, (list, tuple)):
                            if len(donor) != len(baseline_entity_features_full):
                                raise ValueError(
                                    f"ssc_state_override[{role_name}] list length {len(donor)} does not match "
                                    f"batch size {len(baseline_entity_features_full)}"
                                )
                            donor = donor[scene_idx]
                        if donor is None:
                            continue
                        if not torch.is_tensor(donor):
                            donor = torch.as_tensor(donor)
                        donor = donor.to(device=state_value.device, dtype=state_value.dtype)
                        if donor.ndim == 3:
                            if int(donor.shape[0]) != 1:
                                raise ValueError(
                                    f"ssc_state_override[{role_name}] per-scene donor must have shape "
                                    "[slots, dim] or [1, slots, dim]"
                                )
                            donor = donor[0]
                        if donor.ndim != 2:
                            raise ValueError(
                                f"ssc_state_override[{role_name}] must have shape [slots, dim]"
                            )
                        if tuple(donor.shape) != tuple(state_value.shape):
                            raise ValueError(
                                f"ssc_state_override[{role_name}] shape mismatch: got {tuple(donor.shape)}, "
                                f"expected {tuple(state_value.shape)}"
                            )
                        if role_name == "S_ent":
                            S_ent = donor
                        elif role_name == "S_frame":
                            S_frame = donor
                        else:
                            S_rel = donor
                        override_roles.append(role_name)

                    if override_roles and use_scene:
                        pooled_ent = S_ent.mean(dim=0, keepdim=True)
                        pooled_frame = S_frame.mean(dim=0, keepdim=True)
                        pooled_rel = S_rel.mean(dim=0, keepdim=True)
                        pooled_scene = torch.cat([pooled_ent, pooled_frame, pooled_rel], dim=-1)
                        scene_proj = self.get_model().pc_ssc_scene_proj
                        scene_raw = scene_proj(
                            pooled_scene.to(device=scene_proj.weight.device, dtype=scene_proj.weight.dtype)
                        ).to(device=S_ent.device, dtype=S_ent.dtype)
                        S_scene = scene_raw.view(max(scene_summary_token_count, 1), pc_dim)

                ssc_state_override_records.append({
                    "scene_index": int(scene_idx),
                    "applied": bool(override_roles),
                    "roles": list(override_roles),
                    "scene_recomputed": bool(override_roles and use_scene),
                    "attention_is_original_construction_path": bool(override_roles),
                })

                ssc_used_frame_evidence_counts.append(int(frame_state_evidence_count))

                if detail_tokens > 0:
                    detail_pc = baseline_pc[:min(detail_tokens, int(baseline_pc.shape[0]))]
                    if int(detail_pc.shape[0]) < detail_tokens:
                        pad_count = int(detail_tokens - detail_pc.shape[0])
                        detail_pc = torch.cat([detail_pc, baseline_pc.new_zeros((pad_count, pc_dim))], dim=0)
                else:
                    detail_pc = baseline_pc.new_zeros((0, pc_dim))

                detail_mm = self.get_model().mm_projector(detail_pc)
                if bool(ssc_role_analysis["use_generic_states"]):
                    generic_mm = self.get_model().mm_projector(S_generic)
                    generic_embed = self.get_model().pc_ssc_generic_token_embed
                    generic_mm = generic_mm + generic_embed[0].to(
                        device=generic_mm.device, dtype=generic_mm.dtype
                    )
                    cur_token_parts = [detail_mm, generic_mm]
                else:
                    role_embed = self.get_model().pc_ssc_token_role_embed
                    detail_mm = detail_mm + role_embed[0].to(device=detail_mm.device, dtype=detail_mm.dtype)
                    cur_token_parts = [detail_mm]
                    if use_ent:
                        ent_mm = self.get_model().mm_projector(S_ent)
                        ent_mm = ent_mm + role_embed[1].to(device=ent_mm.device, dtype=ent_mm.dtype)
                        cur_token_parts.append(ent_mm)
                    if use_frame:
                        frame_mm = self.get_model().mm_projector(S_frame)
                        frame_mm = frame_mm + role_embed[2].to(device=frame_mm.device, dtype=frame_mm.dtype)
                        cur_token_parts.append(frame_mm)
                    if use_rel:
                        rel_mm = self.get_model().mm_projector(S_rel)
                        rel_mm = rel_mm + role_embed[3].to(device=rel_mm.device, dtype=rel_mm.dtype)
                        cur_token_parts.append(rel_mm)
                    if use_scene:
                        scene_mm = self.get_model().mm_projector(S_scene)
                        scene_mm = scene_mm + role_embed[4].to(device=scene_mm.device, dtype=scene_mm.dtype)
                        cur_token_parts.append(scene_mm)

                cur_tokens = torch.cat(cur_token_parts, dim=0)
                final_token_count_match = final_token_count_match and (int(cur_tokens.shape[0]) == int(num_pc_tokens))
                if int(cur_tokens.shape[0]) != int(num_pc_tokens):
                    raise RuntimeError(
                        f"SSC3D final token count mismatch: got {int(cur_tokens.shape[0])}, expected {int(num_pc_tokens)}."
                    )
                pointcloud_tokens.append(cur_tokens)

                if bool(ssc_role_analysis["frame_cov_loss_enabled"]) and use_frame:
                    frame_cov_loss = self.get_model()._compute_ssc_frame_coverage_loss(attn_frame, frame_state_group_ids_valid)
                    if frame_cov_loss is None:
                        frame_cov_loss = zero_loss
                else:
                    frame_cov_loss = zero_loss
                relation_boundary_logits = None
                relation_region_logits = None
                relation_boundary_label = None
                relation_region_label = None
                if bool(ssc_role_analysis["rel_geo_loss_enabled"]) and use_rel:
                    rel_boundary_loss, rel_region_loss = self.get_model()._compute_ssc_relation_geo_loss(
                        S_rel,
                        entity_primitive_xyz,
                        sp_xyz,
                        entity_primitive_weights,
                    )
                    if int(S_rel.shape[0]) > 0 and int(entity_primitive_xyz.shape[0]) > 0:
                        if (
                            entity_primitive_weights is not None
                            and torch.is_tensor(entity_primitive_weights)
                            and int(entity_primitive_weights.numel()) == int(entity_primitive_xyz.shape[0])
                        ):
                            centroid_weights = entity_primitive_weights.float().view(-1)
                            centroid_weights = centroid_weights / centroid_weights.sum().clamp_min(1e-6)
                            relation_centroid = (
                                entity_primitive_xyz.float() * centroid_weights.unsqueeze(-1)
                            ).sum(dim=0)
                        else:
                            relation_centroid = entity_primitive_xyz.float().mean(dim=0)
                        scene_xyz = sp_xyz.float()
                        xyz_min = scene_xyz.min(dim=0)[0]
                        xyz_max = scene_xyz.max(dim=0)[0]
                        boundary_dist = torch.stack(
                            [
                                relation_centroid[0] - xyz_min[0],
                                xyz_max[0] - relation_centroid[0],
                                relation_centroid[1] - xyz_min[1],
                                xyz_max[1] - relation_centroid[1],
                            ],
                            dim=0,
                        )
                        relation_boundary_label = boundary_dist.argmin().view(1).to(torch.long)
                        grid_x = max(int(getattr(self.config, "dvc_region_grid_x", 3)), 1)
                        grid_y = max(int(getattr(self.config, "dvc_region_grid_y", 3)), 1)
                        xy_span = (xyz_max[:2] - xyz_min[:2]).clamp_min(1e-6)
                        norm_xy = ((relation_centroid[:2] - xyz_min[:2]) / xy_span).clamp(0.0, 1.0 - 1e-6)
                        cell_x = torch.floor(norm_xy[0] * grid_x).long().clamp(0, grid_x - 1)
                        cell_y = torch.floor(norm_xy[1] * grid_y).long().clamp(0, grid_y - 1)
                        relation_region_label = (cell_y * grid_x + cell_x).view(1).to(torch.long)
                        pooled_rel = S_rel.mean(dim=0, keepdim=True)
                        relation_boundary_logits = self.get_model().pc_ssc_rel_boundary_head(
                            pooled_rel.to(dtype=self.get_model().pc_ssc_rel_boundary_head.weight.dtype)
                        )
                        relation_region_logits = self.get_model().pc_ssc_rel_region_head(
                            pooled_rel.to(dtype=self.get_model().pc_ssc_rel_region_head.weight.dtype)
                        )
                else:
                    rel_boundary_loss = zero_loss
                    rel_region_loss = zero_loss
                if bool(ssc_role_analysis["ent_sem_loss_enabled"]) and use_ent:
                    ent_sem_loss = self.get_model()._compute_ssc_entity_semantic_loss(
                        S_ent,
                        entity_state_evidence,
                        entity_state_weights,
                    )
                else:
                    ent_sem_loss = zero_loss
                ssc_frame_cov_losses.append(frame_cov_loss)
                ssc_rel_boundary_losses.append(rel_boundary_loss)
                ssc_rel_region_losses.append(rel_region_loss)
                ssc_ent_sem_losses.append(ent_sem_loss)

                ssc_state_ent.append(S_ent)
                ssc_state_frame.append(S_frame)
                ssc_state_rel.append(S_rel)
                ssc_state_scene.append(S_scene)
                ssc_state_generic.append(S_generic)
                ssc_ent_state_norm.append(
                    float(S_ent.norm(dim=-1).mean().item()) if use_ent and int(S_ent.shape[0]) > 0 else 0.0
                )
                ssc_frame_state_norm.append(
                    float(S_frame.norm(dim=-1).mean().item()) if use_frame and int(S_frame.shape[0]) > 0 else 0.0
                )
                ssc_rel_state_norm.append(
                    float(S_rel.norm(dim=-1).mean().item()) if use_rel and int(S_rel.shape[0]) > 0 else 0.0
                )
                ssc_generic_state_norm.append(
                    float(S_generic.norm(dim=-1).mean().item()) if int(S_generic.shape[0]) > 0 else 0.0
                )
                if capture_ssc_state_vis:
                    entity_attn_scores = (
                        attn_ent.float().mean(dim=0)
                        if attn_ent is not None and int(attn_ent.numel()) > 0
                        else None
                    )
                    frame_attn_scores = (
                        attn_frame.float().mean(dim=0)
                        if attn_frame is not None and int(attn_frame.numel()) > 0
                        else None
                    )
                    relation_entity_scores = None
                    relation_frame_scores = None
                    relation_entity_slot_scores = None
                    relation_frame_slot_scores = None
                    if attn_rel is not None and int(attn_rel.numel()) > 0:
                        rel_source_scores = attn_rel.float().mean(dim=0)
                        if ssc_role_analysis["relation_source"] == "primitive_evidence":
                            num_entity_sources = int(entity_primitives.shape[0])
                            relation_entity_scores = rel_source_scores[:num_entity_sources]
                            relation_frame_scores = rel_source_scores[
                                num_entity_sources:num_entity_sources + int(frame_state_evidence.shape[0])
                            ]
                        else:
                            num_ent_slots = int(S_ent.shape[0])
                            num_frame_slots = int(S_frame.shape[0])
                            relation_entity_slot_scores = rel_source_scores[:num_ent_slots]
                            relation_frame_slot_scores = rel_source_scores[
                                num_ent_slots:num_ent_slots + num_frame_slots
                            ]
                            relation_entity_scores = _weighted_state_to_evidence(
                                relation_entity_slot_scores,
                                attn_ent,
                            )
                            relation_frame_scores = _weighted_state_to_evidence(
                                relation_frame_slot_scores,
                                attn_frame,
                            )
                    if entity_attn_scores is not None and int(entity_primitive_xyz.shape[0]) != int(entity_attn_scores.shape[0]):
                        entity_attn_scores = None
                    if frame_attn_scores is not None and int(frame_state_evidence_xyz.shape[0]) != int(frame_attn_scores.shape[0]):
                        frame_attn_scores = None
                    if relation_entity_scores is not None and int(entity_primitive_xyz.shape[0]) != int(relation_entity_scores.shape[0]):
                        relation_entity_scores = None
                    if relation_frame_scores is not None and int(frame_state_evidence_xyz.shape[0]) != int(relation_frame_scores.shape[0]):
                        relation_frame_scores = None

                    ssc_state_vis_records.append(
                        {
                            "scene_index": int(scene_idx),
                            "entity_evidence_source": str(ssc_role_analysis["entity_evidence_source"]),
                            "frame_evidence_source": str(ssc_role_analysis["frame_evidence_source"]),
                            "relation_source": str(ssc_role_analysis["relation_source"]),
                            "entity_evidence_xyz": _tensor_to_cpu_float(entity_primitive_xyz),
                            "entity_evidence_scores": _tensor_to_cpu_float(entity_attn_scores),
                            "entity_selection_scores": _tensor_to_cpu_float(entity_primitive_weights),
                            "entity_topk_indices": _tensor_to_cpu_long(entity_topk_idx),
                            "entity_all_scores": _tensor_to_cpu_float(entity_all_scores),
                            "entity_candidate_features": _tensor_to_cpu_float(sp_feat),
                            "entity_candidate_xyz": _tensor_to_cpu_float(sp_xyz),
                            "state_ent": _tensor_to_cpu_float(S_ent),
                            "state_frame": _tensor_to_cpu_float(S_frame),
                            "state_rel": _tensor_to_cpu_float(S_rel),
                            "state_scene": _tensor_to_cpu_float(S_scene),
                            "state_override_roles": list(override_roles),
                            "state_override_scene_recomputed": bool(override_roles and use_scene),
                            "state_override_attention_is_original": bool(override_roles),
                            "residual_features": _tensor_to_cpu_float(detail_pc),
                            "frame_anchor_coord": _tensor_to_cpu_float(frame_anchor_coord_valid),
                            "attention_ent_heads": _tensor_to_cpu_float(attn_ent),
                            "attention_frame_heads": _tensor_to_cpu_float(attn_frame),
                            "attention_rel_heads": _tensor_to_cpu_float(attn_rel),
                            "frame_evidence_xyz": _tensor_to_cpu_float(frame_state_evidence_xyz),
                            "frame_evidence_scores": _tensor_to_cpu_float(frame_attn_scores),
                            "frame_group_ids": [int(v) for v in frame_state_group_ids_valid],
                            "frame_group_kind": [int(v) for v in frame_state_group_kind_valid],
                            "relation_entity_evidence_xyz": _tensor_to_cpu_float(entity_primitive_xyz),
                            "relation_entity_evidence_scores": _tensor_to_cpu_float(relation_entity_scores),
                            "relation_frame_evidence_xyz": _tensor_to_cpu_float(frame_state_evidence_xyz),
                            "relation_frame_evidence_scores": _tensor_to_cpu_float(relation_frame_scores),
                            "relation_entity_slot_scores": _tensor_to_cpu_float(relation_entity_slot_scores),
                            "relation_frame_slot_scores": _tensor_to_cpu_float(relation_frame_slot_scores),
                            "relation_boundary_logits": _tensor_to_cpu_float(relation_boundary_logits),
                            "relation_region_logits": _tensor_to_cpu_float(relation_region_logits),
                            "relation_boundary_label": _tensor_to_cpu_long(relation_boundary_label),
                            "relation_region_label": _tensor_to_cpu_long(relation_region_label),
                        }
                    )

            if profile_ssc_constructor:
                ssc_constructor_end.record()
                ssc_constructor_end.synchronize()
                mask_input_dict["ssc_constructor_cuda_ms"] = float(ssc_constructor_start.elapsed_time(ssc_constructor_end))

            if len(ssc_frame_cov_losses) > 0:
                frame_cov_loss = torch.stack([loss.float() for loss in ssc_frame_cov_losses]).mean()
                rel_boundary_loss = torch.stack([loss.float() for loss in ssc_rel_boundary_losses]).mean()
                rel_region_loss = torch.stack([loss.float() for loss in ssc_rel_region_losses]).mean()
                ent_sem_loss = torch.stack([loss.float() for loss in ssc_ent_sem_losses]).mean()
            else:
                aggregate_zero_loss = self.get_model().pc_ssc_seg_alpha.float().sum() * 0.0
                frame_cov_loss = aggregate_zero_loss
                rel_boundary_loss = aggregate_zero_loss
                rel_region_loss = aggregate_zero_loss
                ent_sem_loss = aggregate_zero_loss

            total_aux_loss = (
                float(getattr(self.config, "ssc_frame_cov_loss_weight", 0.05)) * frame_cov_loss
                + float(getattr(self.config, "ssc_rel_boundary_loss_weight", 0.05)) * rel_boundary_loss
                + float(getattr(self.config, "ssc_rel_region_loss_weight", 0.05)) * rel_region_loss
                + float(getattr(self.config, "ssc_ent_sem_loss_weight", 0.05)) * ent_sem_loss
            )

            boundary_selected_counts = [int(v) for v in mask_input_dict.get("boundary_selected_counts", [0] * len(frame_token_counts))]
            region_selected_counts = [int(v) for v in mask_input_dict.get("region_selected_counts", [0] * len(frame_token_counts))]
            if len(boundary_selected_counts) < len(frame_token_counts):
                boundary_selected_counts = boundary_selected_counts + [0] * (len(frame_token_counts) - len(boundary_selected_counts))
            if len(region_selected_counts) < len(frame_token_counts):
                region_selected_counts = region_selected_counts + [0] * (len(frame_token_counts) - len(region_selected_counts))
            boundary_selected_counts = boundary_selected_counts[:len(frame_token_counts)]
            region_selected_counts = region_selected_counts[:len(frame_token_counts)]

            prompt_tokens = [self.get_model().mm_projector(feat) for feat in prompt_features]
            final_token_counts = [int(tok.shape[0]) for tok in pointcloud_tokens]
            mask_input_dict["entity_token_counts"] = entity_token_counts
            mask_input_dict["frame_token_counts"] = frame_token_counts
            mask_input_dict["frame_valid_counts"] = frame_valid_counts
            mask_input_dict["boundary_selected_counts"] = boundary_selected_counts
            mask_input_dict["region_selected_counts"] = region_selected_counts
            mask_input_dict["ssc_frame_evidence_counts_used"] = ssc_used_frame_evidence_counts
            mask_input_dict["ssc_state_bank"] = {
                "S_ent": ssc_state_ent,
                "S_frame": ssc_state_frame,
                "S_rel": ssc_state_rel,
                "S_scene": ssc_state_scene,
                "S_generic": ssc_state_generic,
            }
            mask_input_dict["ssc_state_override"] = ssc_state_override_records
            mask_input_dict["ssc_aux_losses"] = {
                "frame_cov": frame_cov_loss,
                "rel_boundary": rel_boundary_loss,
                "rel_region": rel_region_loss,
                "ent_sem": ent_sem_loss,
                "total": total_aux_loss,
            }
            ssc_debug = {
                "ssc_ablation_mode": str(ssc_layout["ablation_mode"]),
                "ssc_role_analysis_variant": str(ssc_role_analysis["variant"]),
                "detail_tokens": int(detail_tokens),
                "ssc_detail_tokens_resolved": ssc_detail_tokens_resolved,
                "ssc_baseline_entity_request_tokens": ssc_baseline_entity_request_tokens,
                "original_ent_slots": ent_slots,
                "original_frame_slots": frame_slots,
                "original_rel_slots": rel_slots,
                "generic_state_slots": int(generic_state_slots),
                "effective_ent_slots": int(ssc_layout["effective_ent_slots"]),
                "effective_frame_slots": int(ssc_layout["effective_frame_slots"]),
                "effective_rel_slots": int(ssc_layout["effective_rel_slots"]),
                "effective_scene_summary_token_count": int(ssc_layout["effective_scene_summary_token_count"]),
                "entity_candidate_topk": int(entity_candidate_topk),
                "entity_selector": str(entity_selector),
                "selector_seed": int(selector_seed),
                "state_override": ssc_state_override_records,
                "state_override_applied": bool(any(record["applied"] for record in ssc_state_override_records)),
                "state_override_attention_is_original_construction_path": bool(
                    any(record["attention_is_original_construction_path"] for record in ssc_state_override_records)
                ),
                "frame_primitive_budget": frame_primitive_budget,
                "frame_object_token_counts": ssc_frame_object_token_counts,
                "frame_evidence_counts_used": ssc_used_frame_evidence_counts,
                "frame_token_counts": frame_token_counts,
                "frame_valid_counts": frame_valid_counts,
                "boundary_selected_counts": boundary_selected_counts,
                "region_selected_counts": region_selected_counts,
                "ent_state_norm_mean": float(sum(ssc_ent_state_norm) / float(max(len(ssc_ent_state_norm), 1))),
                "frame_state_norm_mean": float(sum(ssc_frame_state_norm) / float(max(len(ssc_frame_state_norm), 1))),
                "rel_state_norm_mean": float(sum(ssc_rel_state_norm) / float(max(len(ssc_rel_state_norm), 1))),
                "generic_state_norm_mean": float(sum(ssc_generic_state_norm) / float(max(len(ssc_generic_state_norm), 1))),
                "frame_cov_loss": float(frame_cov_loss.detach().float().item()),
                "rel_boundary_loss": float(rel_boundary_loss.detach().float().item()),
                "rel_region_loss": float(rel_region_loss.detach().float().item()),
                "ent_sem_loss": float(ent_sem_loss.detach().float().item()),
                "scene_summary_token_count": int(scene_summary_token_count if not bool(ssc_role_analysis["use_generic_states"]) else 0),
                "detail_dropout_rate": float(getattr(self.config, "ssc_detail_dropout_rate", 0.2)),
                "entity_evidence_source": str(ssc_role_analysis["entity_evidence_source"]),
                "frame_evidence_source": str(ssc_role_analysis["frame_evidence_source"]),
                "relation_source": str(ssc_role_analysis["relation_source"]),
                "frame_cov_loss_enabled": bool(ssc_role_analysis["frame_cov_loss_enabled"]),
                "rel_geo_loss_enabled": bool(ssc_role_analysis["rel_geo_loss_enabled"]),
                "ent_sem_loss_enabled": bool(ssc_role_analysis["ent_sem_loss_enabled"]),
                "seg_role_bias_enabled": bool(ssc_role_analysis["seg_role_bias_enabled"]),
                "role_analysis_notes": list(ssc_role_analysis["notes"]),
                # Current SSC3D relation geo loss supervises coarse weighted entity-centroid geometry.
                "rel_geo_uses_weighted_centroid": bool(use_rel and ssc_role_analysis["rel_geo_loss_enabled"]),
                "rel_geo_supervision_mode": (
                    "weighted_entity_centroid"
                    if bool(use_rel and ssc_role_analysis["rel_geo_loss_enabled"])
                    else "disabled"
                ),
                "geometry_source_valid": bool(use_rel and ssc_role_analysis["rel_geo_loss_enabled"]),
                "expected_total_tokens": int(ssc_layout["expected_total_tokens"]),
                "final_token_counts": final_token_counts,
                "final_token_count_match": bool(final_token_count_match),
            }
            mask_input_dict["ssc_debug"] = ssc_debug
            self.get_model()._last_ssc_debug = ssc_debug
            if capture_ssc_state_vis:
                mask_input_dict["ssc_state_vis"] = ssc_state_vis_records
                self.get_model()._last_ssc_state_vis = ssc_state_vis_records
            mask_input_dict["dvc_debug"] = {
                "dvc_variant": dvc_variant,
                "dvc_second_token_semantics": configured_second_semantics,
                "ssc_ablation_mode": str(ssc_layout["ablation_mode"]),
                "ssc_role_analysis_variant": str(ssc_role_analysis["variant"]),
                "entity_token_counts": entity_token_counts,
                "frame_token_counts": frame_token_counts,
                "frame_valid_counts": frame_valid_counts,
                "frame_evidence_counts_used": ssc_used_frame_evidence_counts,
                "boundary_selected_counts": boundary_selected_counts,
                "region_selected_counts": region_selected_counts,
                "ent_state_norm_mean": ssc_debug["ent_state_norm_mean"],
                "frame_state_norm_mean": ssc_debug["frame_state_norm_mean"],
                "rel_state_norm_mean": ssc_debug["rel_state_norm_mean"],
                "generic_state_norm_mean": ssc_debug["generic_state_norm_mean"],
                "final_token_counts": final_token_counts,
                "final_token_count_match": bool(final_token_count_match),
            }
            self.get_model()._last_dvc_debug = mask_input_dict["dvc_debug"]
            self.get_model()._last_tsp_debug = None
            return pointcloud_tokens, prompt_tokens, superpoint_features, mask_input_dict

        elif getattr(self.config, "dvc_enable", False):
            self.get_model()._maybe_initialize_pc_dvc_modules(self.config.pc_hidden_size)
            layout_features = frame_anchor_features
            rel_anchor_features = frame_anchor_features

            entity_proj_shared = [self.get_model().mm_projector(feat) for feat in baseline_entity_features_full]
            baseline_tokens = [tok.clone() for tok in entity_proj_shared]

            ctx_res = layout_features
            if getattr(self.config, "dvc_use_layout_adapter", True):
                ctx_res = [self.get_model().pc_layout_adapter(feat, gate_scale=gate_scale) for feat in layout_features]
            ctx_proj_shared = [self.get_model().mm_projector(feat) for feat in ctx_res]
            ra_res = rel_anchor_features
            if getattr(self.config, "dvc_use_layout_adapter", True):
                ra_res = [self.get_model().pc_layout_adapter(feat, gate_scale=gate_scale) for feat in rel_anchor_features]
            ra_proj_shared = [self.get_model().mm_projector(feat) for feat in ra_res]
            if getattr(self.config, "dvc_use_tail_role_heads", True):
                ctx_proj = [self.get_model().pc_layout_role_head(feat, gate_scale=gate_scale) for feat in ctx_proj_shared]
                ra_proj = [self.get_model().pc_layout_role_head(feat, gate_scale=gate_scale) for feat in ra_proj_shared]
            else:
                ctx_proj = ctx_proj_shared
                ra_proj = ra_proj_shared

            pointcloud_tokens = []
            final_token_count_match = True
            default_second_semantics = str(getattr(self.config, "dvc_second_token_semantics", "generic_layout"))
            ra_num_slots = max(int(getattr(self.config, "dvc_ra_num_slots", 4)), 0)
            for baseline_tok, entity_proj, ctx_tok, ra_tok, scene_meta in zip(
                baseline_tokens,
                entity_proj_shared,
                ctx_proj,
                ra_proj,
                mask_input_dict["dvc_meta"],
            ):
                keep_count = min(int(scene_meta["num_entity_keep"]), entity_proj.shape[0])
                entity_head = entity_proj[:keep_count]
                entity_tail_shared = entity_proj[keep_count:]
                if getattr(self.config, "dvc_use_tail_role_heads", True):
                    entity_tail = self.get_model().pc_entity_tail_role_head(entity_tail_shared, gate_scale=gate_scale)
                else:
                    entity_tail = entity_tail_shared

                used_k_ctx = 0
                used_k_ra = 0
                scene_second_semantics = str(scene_meta.get("second_token_semantics", default_second_semantics))
                if scene_second_semantics == "relation_anchor":
                    used_k_ctx = 0
                    used_k_ra = min(int(ra_tok.shape[0]), int(entity_tail.shape[0]))
                    second_tok = ra_tok
                elif scene_second_semantics == "hybrid_ra":
                    k_rel = int(entity_tail.shape[0])
                    k_ra = min(ra_num_slots, int(ra_tok.shape[0]), k_rel)
                    k_ctx = min(int(ctx_tok.shape[0]), max(k_rel - k_ra, 0))
                    used_k_ra = int(k_ra)
                    used_k_ctx = int(k_ctx)
                    second_parts = []
                    if k_ctx > 0:
                        second_parts.append(ctx_tok[:k_ctx])
                    if k_ra > 0:
                        second_parts.append(ra_tok[:k_ra])
                    if len(second_parts) > 0:
                        second_tok = torch.cat(second_parts, dim=0)
                    else:
                        second_tok = entity_tail.new_zeros((0, entity_tail.shape[-1]))
                else:
                    used_k_ctx = min(int(ctx_tok.shape[0]), int(entity_tail.shape[0]))
                    used_k_ra = 0
                    second_tok = ctx_tok

                used_ctx_second_counts.append(int(used_k_ctx))
                used_ra_second_counts.append(int(used_k_ra))
                cur_second_used = int(used_k_ctx + used_k_ra)
                cur_second_denom = max(int(entity_tail.shape[0]), 1)
                second_path_used_counts.append(cur_second_used)
                second_path_fill_ratio.append(float(cur_second_used) / float(cur_second_denom))
                ra_slot_ratio.append(float(used_k_ra) / float(cur_second_denom))

                tail_layout_target = entity_tail.clone()
                effective_layout_slots = min(second_tok.shape[0], entity_tail.shape[0])
                if effective_layout_slots > 0:
                    tail_layout_target[:effective_layout_slots] = second_tok[:effective_layout_slots]

                if getattr(self.config, "dvc_use_layout_mix_gate", True):
                    mixed_tail = entity_tail.clone()
                    if effective_layout_slots > 0:
                        if getattr(self.config, "dvc_use_slotwise_layout_mix", True):
                            slot_delta = self.get_model().pc_layout_slot_gate(
                                entity_tail[:effective_layout_slots],
                                tail_layout_target[:effective_layout_slots],
                            )
                        else:
                            slot_delta = torch.zeros(
                                (effective_layout_slots, 1),
                                device=entity_tail.device,
                                dtype=entity_tail.dtype,
                            )
                        raw_gate = self.get_model().pc_layout_mix_gate.view(1, 1) + slot_delta
                        effective_gate = gate_scale * torch.tanh(raw_gate)
                        mixed_prefix = entity_tail[:effective_layout_slots] + effective_gate * (
                            tail_layout_target[:effective_layout_slots] - entity_tail[:effective_layout_slots]
                        )
                        mixed_tail[:effective_layout_slots] = mixed_prefix
                else:
                    mixed_tail = entity_tail

                if getattr(self.config, "dvc_use_role_embed", True):
                    entity_head = entity_head + self.get_model().pc_role_embed[0].to(device=entity_head.device, dtype=entity_head.dtype)
                    final_tail = mixed_tail.clone()
                    if getattr(self.config, "dvc_use_layout_mix_gate", True) and effective_layout_slots > 0:
                        final_tail[:effective_layout_slots] = (
                            final_tail[:effective_layout_slots]
                            + self.get_model().pc_role_embed[1].to(device=final_tail.device, dtype=final_tail.dtype)
                        )
                else:
                    final_tail = mixed_tail

                cur_tokens = torch.cat([entity_head, final_tail], dim=0)
                final_token_count_match = final_token_count_match and (cur_tokens.shape[0] == baseline_tok.shape[0])
                pointcloud_tokens.append(cur_tokens)

            if (
                getattr(self.config, "dvc_zero_perturbation", True)
                and not getattr(self.get_model(), "_dvc_zero_perturbation_check_done", False)
                and (
                    not getattr(self.config, "dvc_use_layout_adapter", True)
                    or float(self.get_model().pc_layout_adapter.gate.detach().abs().max().item()) == 0.0
                )
                and (
                    not getattr(self.config, "dvc_use_tail_role_heads", True)
                    or float(self.get_model().pc_entity_tail_role_head.gate.detach().abs().max().item()) == 0.0
                )
                and (
                    not getattr(self.config, "dvc_use_tail_role_heads", True)
                    or float(self.get_model().pc_layout_role_head.gate.detach().abs().max().item()) == 0.0
                )
                and (
                    not getattr(self.config, "dvc_use_role_embed", True)
                    or float(self.get_model().pc_role_embed.detach().abs().max().item()) == 0.0
                )
                and (
                    not getattr(self.config, "dvc_use_layout_mix_gate", True)
                    or float(self.get_model().pc_layout_mix_gate.detach().abs().max().item()) == 0.0
                )
                and (
                    not getattr(self.config, "dvc_use_slotwise_layout_mix", True)
                    or (
                        float(self.get_model().pc_layout_slot_gate.fc2.weight.detach().abs().max().item()) == 0.0
                        and float(self.get_model().pc_layout_slot_gate.fc2.bias.detach().abs().max().item()) == 0.0
                    )
                )
            ):
                max_abs_diff = 0.0
                mean_abs_diff = 0.0
                total_items = 0
                for baseline_tok, dvc_tok in zip(baseline_tokens, pointcloud_tokens):
                    if baseline_tok.numel() == 0:
                        continue
                    cur_diff = (baseline_tok - dvc_tok).abs()
                    max_abs_diff = max(max_abs_diff, float(cur_diff.max().item()))
                    mean_abs_diff += float(cur_diff.mean().item())
                    total_items += 1
                if total_items > 0:
                    mean_abs_diff /= total_items
                print(
                    "[DVC zero-perturbation check] "
                    f"final_token_count_match={final_token_count_match}, "
                    f"max_abs_diff={max_abs_diff:.8f}, mean_abs_diff={mean_abs_diff:.8f}"
                )
                self.get_model()._dvc_zero_perturbation_check_done = True
        else:
            pointcloud_tokens = [self.get_model().mm_projector(feat) for feat in baseline_entity_features_full]

        final_token_counts = [int(tok.shape[0]) for tok in pointcloud_tokens]
        raw_layout_token_counts = [int(v) for v in mask_input_dict.get("raw_layout_token_counts", layout_token_counts)]
        raw_relation_token_counts = [int(v) for v in mask_input_dict.get("raw_relation_token_counts", mask_input_dict.get("relation_token_counts", layout_token_counts))]
        boundary_selected_counts = [int(v) for v in mask_input_dict.get("boundary_selected_counts", [0] * len(layout_token_counts))]
        region_selected_counts = [int(v) for v in mask_input_dict.get("region_selected_counts", [0] * len(layout_token_counts))]
        relation_token_counts = [int(v) for v in mask_input_dict.get("relation_token_counts", raw_relation_token_counts)]

        if len(boundary_selected_counts) < len(layout_token_counts):
            boundary_selected_counts = boundary_selected_counts + [0] * (len(layout_token_counts) - len(boundary_selected_counts))
        if len(region_selected_counts) < len(layout_token_counts):
            region_selected_counts = region_selected_counts + [0] * (len(layout_token_counts) - len(region_selected_counts))
        if len(relation_token_counts) < len(layout_token_counts):
            relation_token_counts = relation_token_counts + [0] * (len(layout_token_counts) - len(relation_token_counts))
        if len(raw_layout_token_counts) < len(layout_token_counts):
            raw_layout_token_counts = raw_layout_token_counts + [0] * (len(layout_token_counts) - len(raw_layout_token_counts))
        if len(raw_relation_token_counts) < len(layout_token_counts):
            raw_relation_token_counts = raw_relation_token_counts + [0] * (len(layout_token_counts) - len(raw_relation_token_counts))
        boundary_selected_counts = boundary_selected_counts[:len(layout_token_counts)]
        region_selected_counts = region_selected_counts[:len(layout_token_counts)]
        relation_token_counts = relation_token_counts[:len(layout_token_counts)]
        raw_layout_token_counts = raw_layout_token_counts[:len(layout_token_counts)]
        raw_relation_token_counts = raw_relation_token_counts[:len(layout_token_counts)]
        if len(used_ctx_second_counts) < len(layout_token_counts):
            used_ctx_second_counts = used_ctx_second_counts + [0] * (len(layout_token_counts) - len(used_ctx_second_counts))
        if len(used_ra_second_counts) < len(layout_token_counts):
            used_ra_second_counts = used_ra_second_counts + [0] * (len(layout_token_counts) - len(used_ra_second_counts))
        if len(second_path_used_counts) < len(layout_token_counts):
            second_path_used_counts = second_path_used_counts + [0] * (len(layout_token_counts) - len(second_path_used_counts))
        if len(second_path_fill_ratio) < len(layout_token_counts):
            second_path_fill_ratio = second_path_fill_ratio + [0.0] * (len(layout_token_counts) - len(second_path_fill_ratio))
        if len(ra_slot_ratio) < len(layout_token_counts):
            ra_slot_ratio = ra_slot_ratio + [0.0] * (len(layout_token_counts) - len(ra_slot_ratio))
        used_ctx_second_counts = used_ctx_second_counts[:len(layout_token_counts)]
        used_ra_second_counts = used_ra_second_counts[:len(layout_token_counts)]
        second_path_used_counts = second_path_used_counts[:len(layout_token_counts)]
        second_path_fill_ratio = second_path_fill_ratio[:len(layout_token_counts)]
        ra_slot_ratio = ra_slot_ratio[:len(layout_token_counts)]

        raw_relation_token_ratio = [
            float(rel_count) / float(max(final_count, 1))
            for rel_count, final_count in zip(relation_token_counts, final_token_counts)
        ]
        nonempty_boundary_scene_ratio = (
            float(sum(1 for cnt in boundary_selected_counts if cnt > 0)) / float(len(boundary_selected_counts))
            if len(boundary_selected_counts) > 0 else 0.0
        )
        nonempty_region_scene_ratio = (
            float(sum(1 for cnt in region_selected_counts if cnt > 0)) / float(len(region_selected_counts))
            if len(region_selected_counts) > 0 else 0.0
        )

        dvc_second_token_semantics = str(getattr(self.config, "dvc_second_token_semantics", "generic_layout"))
        if len(mask_input_dict.get("dvc_meta", [])) > 0:
            dvc_second_token_semantics = str(
                mask_input_dict["dvc_meta"][0].get("second_token_semantics", dvc_second_token_semantics)
            )

        if (
            bool(getattr(self.config, "dvc_enable", False))
            and
            dvc_second_token_semantics in {"relation_anchor", "hybrid_ra"}
            and not getattr(self.get_model(), "_dvc_relation_anchor_debug_printed", False)
        ):
            mean_boundary_selected = (
                float(sum(boundary_selected_counts)) / float(len(boundary_selected_counts))
                if len(boundary_selected_counts) > 0 else 0.0
            )
            mean_region_selected = (
                float(sum(region_selected_counts)) / float(len(region_selected_counts))
                if len(region_selected_counts) > 0 else 0.0
            )
            mean_ctx_second_tokens = (
                float(sum(used_ctx_second_counts)) / float(len(used_ctx_second_counts))
                if len(used_ctx_second_counts) > 0 else 0.0
            )
            mean_ra_second_tokens = (
                float(sum(used_ra_second_counts)) / float(len(used_ra_second_counts))
                if len(used_ra_second_counts) > 0 else 0.0
            )
            mean_second_path_fill_ratio = (
                float(sum(second_path_fill_ratio)) / float(len(second_path_fill_ratio))
                if len(second_path_fill_ratio) > 0 else 0.0
            )
            mean_ra_slot_ratio = (
                float(sum(ra_slot_ratio)) / float(len(ra_slot_ratio))
                if len(ra_slot_ratio) > 0 else 0.0
            )
            print(
                "[DVC RSA debug] "
                f"mean_ctx_second_tokens={mean_ctx_second_tokens:.4f}, "
                f"mean_ra_second_tokens={mean_ra_second_tokens:.4f}, "
                f"mean_second_path_fill_ratio={mean_second_path_fill_ratio:.4f}, "
                f"mean_ra_slot_ratio={mean_ra_slot_ratio:.4f}, "
                f"mean_boundary_selected={mean_boundary_selected:.4f}, "
                f"mean_region_selected={mean_region_selected:.4f}, "
                f"final_token_count_match={final_token_count_match}"
            )
            self.get_model()._dvc_relation_anchor_debug_printed = True

        prompt_tokens = [self.get_model().mm_projector(feat) for feat in prompt_features]
        mask_input_dict["entity_token_counts"] = entity_token_counts
        mask_input_dict["layout_token_counts"] = layout_token_counts
        mask_input_dict["raw_layout_token_counts"] = raw_layout_token_counts
        mask_input_dict["boundary_selected_counts"] = boundary_selected_counts
        mask_input_dict["region_selected_counts"] = region_selected_counts
        mask_input_dict["relation_token_counts"] = relation_token_counts
        mask_input_dict["raw_relation_token_counts"] = raw_relation_token_counts
        mask_input_dict["dvc_debug"] = {
            "dvc_enable": bool(getattr(self.config, "dvc_enable", False)),
            "dvc_zero_perturbation": bool(getattr(self.config, "dvc_zero_perturbation", True)),
            "current_dvc_phase": getattr(self.config, "current_dvc_phase", "dvc_warmup"),
            "current_dvc_gate_scale": gate_scale,
            "dvc_use_layout_adapter": bool(getattr(self.config, "dvc_use_layout_adapter", True)),
            "dvc_use_role_embed": bool(getattr(self.config, "dvc_use_role_embed", True)),
            "dvc_use_layout_mix_gate": bool(getattr(self.config, "dvc_use_layout_mix_gate", True)),
            "dvc_use_tail_role_heads": bool(getattr(self.config, "dvc_use_tail_role_heads", True)),
            "dvc_use_slotwise_layout_mix": bool(getattr(self.config, "dvc_use_slotwise_layout_mix", True)),
            "dvc_variant": dvc_variant,
            "dvc_second_token_semantics": dvc_second_token_semantics,
            "dvc_ra_num_slots": int(getattr(self.config, "dvc_ra_num_slots", 4)),
            "entity_token_counts": entity_token_counts,
            "frame_token_counts": frame_token_counts,
            "frame_valid_counts": frame_valid_counts,
            "layout_token_counts": layout_token_counts,
            "raw_layout_token_counts": raw_layout_token_counts,
            "boundary_selected_counts": boundary_selected_counts,
            "region_selected_counts": region_selected_counts,
            "relation_token_counts": relation_token_counts,
            "raw_relation_token_counts": raw_relation_token_counts,
            "raw_relation_token_ratio": raw_relation_token_ratio,
            "num_ctx_second_tokens": used_ctx_second_counts,
            "num_ra_second_tokens": used_ra_second_counts,
            "second_path_used_counts": second_path_used_counts,
            "second_path_fill_ratio": second_path_fill_ratio,
            "ra_slot_ratio": ra_slot_ratio,
            "nonempty_boundary_scene_ratio": nonempty_boundary_scene_ratio,
            "nonempty_region_scene_ratio": nonempty_region_scene_ratio,
            "final_token_counts": final_token_counts,
            "final_token_count_match": bool(final_token_count_match),
        }
        self.get_model()._last_dvc_debug = mask_input_dict["dvc_debug"]
        if dvc_variant != "tsp3d":
            self.get_model()._last_tsp_debug = None
    
        return pointcloud_tokens, prompt_tokens, superpoint_features, mask_input_dict

    def encode_click_prompt(self, prompt):
        prompt_tokens = self.get_model().get_prompt_encoder()(prompt)
        return prompt_tokens

    def encode_inst_prompt(self, feature):
        if self.config.mm_inst_prompt_encoder == "shared_projector":
            # prompt_tokens = self.get_model().pointcloud_tower.alignment_proj(feature)
            prompt_tokens = self.get_model().mm_projector(feature)
        else:
            prompt_tokens = self.get_model().get_inst_prompt_encoder()(feature)
        return prompt_tokens
        
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values,
        labels=None, coord=None, grid_coord=None, offset=None, pc_input=None,
        p2v_map=None, v2p_map=None, spatial_shape=None, superpoint_mask=None,
        click_mask=None, entity_selector="learned", selector_seed=0, selector_scene_id=None,
        ssc_state_override=None,
    ):
        pointcloud_tower = self.get_pointcloud_tower()
        if pointcloud_tower is None or pc_input is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None
        
        pc_tokens, prompt_tokens, superpoint_features, mask_input_dict = self.encode_pointclouds(
             coord, grid_coord, offset, pc_input, p2v_map, v2p_map, spatial_shape, superpoint_mask, click_mask,
             entity_selector=entity_selector, selector_seed=selector_seed,
             selector_scene_id=selector_scene_id,
             ssc_state_override=ssc_state_override,
        )

        prompt_tokens = torch.cat(prompt_tokens, dim=0)
        track_ssc_detail_dropout = str(getattr(self.config, "dvc_variant", "legacy")) == "ssc3d"
        detail_dropout_token_counts = [0 for _ in pc_tokens] if track_ssc_detail_dropout else []
        detail_dropout_applied_counts = [0 for _ in pc_tokens] if track_ssc_detail_dropout else []
        detail_keep_ratio_list = [1.0 for _ in pc_tokens] if track_ssc_detail_dropout else []

        # get segmentation token index, shift 1 token since the prediciton is the next token
        seg_token_mask = input_ids[:, 1:] == self.config.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        # remove the padding using attention mask
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        seg_token_mask = [cur_seg_mask[cur_attention_mask] for cur_seg_mask, cur_attention_mask in zip(seg_token_mask, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_seg_token_mask = []
        # Use "image" to represent "3d scene" in the following code, following the practice of LLaVA
        # Remember that in the current implementation, the <pc> also uses IMAGE_TOKEN_INDEX
        # May by modified in the future version
        cur_image_idx = 0
        cur_prompt_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_prompts = (cur_input_ids == LOC_TOKEN_INDEX).sum()
            num_specials = num_images + num_prompts
            if num_images == 0:
                cur_pc_features = pc_tokens[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # it seems that the following code has not actually invovled cur_pc_featurs?
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_pc_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            prompt_token_indices = torch.where(cur_input_ids == LOC_TOKEN_INDEX)[0].tolist()
            special_token_indices = sorted(image_token_indices + prompt_token_indices)
            special_tokens = [cur_input_ids[indice] for indice in special_token_indices]
            special_token_indices = [-1] + special_token_indices + [cur_input_ids.shape[0]]

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            cur_seg_token_mask = seg_token_mask[batch_idx]
            cur_seg_token_mask_noim = []
            for i in range(len(special_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[special_token_indices[i]+1:special_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[special_token_indices[i]+1:special_token_indices[i+1]])
                cur_seg_token_mask_noim.append(cur_seg_token_mask[special_token_indices[i]+1:special_token_indices[i+1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_seg_token_mask = []
            for i in range(num_specials + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_seg_token_mask.append(cur_seg_token_mask_noim[i])
                if i < num_specials:
                    cur_token = special_tokens[i]
                    if cur_token == IMAGE_TOKEN_INDEX:
                        scene_idx = cur_image_idx
                        cur_pc_features = pc_tokens[cur_image_idx]
                        cur_image_idx += 1
                        detail_dropout_rate = max(
                            min(float(getattr(self.config, "ssc_detail_dropout_rate", 0.0)), 1.0),
                            0.0,
                        )
                        if (
                            self.training
                            and str(getattr(self.config, "dvc_variant", "legacy")) == "ssc3d"
                            and detail_dropout_rate > 0.0
                            and not bool(cur_seg_token_mask.any().item())
                        ):
                            detail_counts = mask_input_dict.get("ssc_detail_tokens_resolved", [])
                            detail_token_count = int(detail_counts[scene_idx]) if scene_idx < len(detail_counts) else 0
                            detail_token_count = min(max(detail_token_count, 0), int(cur_pc_features.shape[0]))
                            if scene_idx < len(detail_dropout_token_counts):
                                detail_dropout_token_counts[scene_idx] = detail_token_count
                            if detail_token_count > 0:
                                detail_keep = (
                                    torch.rand((detail_token_count,), device=cur_pc_features.device)
                                    >= detail_dropout_rate
                                )
                                kept_count = int(detail_keep.sum().item())
                                dropped_count = int(detail_token_count - kept_count)
                                if scene_idx < len(detail_dropout_applied_counts):
                                    detail_dropout_applied_counts[scene_idx] = dropped_count
                                    detail_keep_ratio_list[scene_idx] = float(kept_count) / float(max(detail_token_count, 1))
                                cur_pc_features = cur_pc_features.clone()
                                cur_pc_features[:detail_token_count] = (
                                    cur_pc_features[:detail_token_count]
                                    * detail_keep.to(dtype=cur_pc_features.dtype).unsqueeze(-1)
                                )
                        cur_new_input_embeds.append(cur_pc_features)
                        cur_new_labels.append(torch.full((cur_pc_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_seg_token_mask.append(torch.full((cur_pc_features.shape[0],), False, device=cur_seg_token_mask.device, dtype=cur_seg_token_mask.dtype))
                    elif cur_token == LOC_TOKEN_INDEX:
                        cur_prompt_features = prompt_tokens[cur_prompt_idx]
                        if len(cur_prompt_features.shape) == 1:
                            cur_prompt_features = cur_prompt_features.unsqueeze(0)
                        cur_prompt_idx += 1
                        cur_new_input_embeds.append(cur_prompt_features)
                        cur_new_labels.append(torch.full((cur_prompt_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_seg_token_mask.append(torch.full((cur_prompt_features.shape[0],), False, device=cur_seg_token_mask.device, dtype=cur_seg_token_mask.dtype))


            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_seg_token_mask = torch.cat(cur_new_seg_token_mask)

            if num_prompts == 0:
                cur_new_input_embeds = torch.cat([cur_new_input_embeds, prompt_tokens[0:0]], dim=0)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_seg_token_mask.append(cur_new_seg_token_mask)

        if track_ssc_detail_dropout:
            detail_keep_ratio_mean = (
                float(sum(detail_keep_ratio_list)) / float(max(len(detail_keep_ratio_list), 1))
            )
            mask_input_dict["detail_dropout_token_counts"] = detail_dropout_token_counts
            mask_input_dict["detail_dropout_applied_counts"] = detail_dropout_applied_counts
            mask_input_dict["detail_keep_ratio_list"] = detail_keep_ratio_list
            ssc_debug = mask_input_dict.get("ssc_debug", {})
            ssc_debug["scene_summary_token_count"] = int(ssc_debug.get("scene_summary_token_count", 1))
            ssc_debug["detail_dropout_token_counts"] = detail_dropout_token_counts
            ssc_debug["detail_dropout_applied_counts"] = detail_dropout_applied_counts
            ssc_debug["detail_keep_ratio_list"] = detail_keep_ratio_list
            ssc_debug["detail_keep_ratio_mean"] = detail_keep_ratio_mean
            mask_input_dict["ssc_debug"] = ssc_debug
            self.get_model()._last_ssc_debug = ssc_debug

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_seg_token_mask = [x[:tokenizer_model_max_length] for x in new_seg_token_mask]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        seg_token_mask_padded = torch.full((batch_size, max_len), False, dtype=new_seg_token_mask[0].dtype, device=new_seg_token_mask[0].device)

        for i, (cur_new_embed, cur_new_labels, cur_new_seg_token_mask) in \
                    enumerate(zip(new_input_embeds, new_labels, new_seg_token_mask)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    seg_token_mask_padded[i, -cur_len:] = cur_new_seg_token_mask
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    seg_token_mask_padded[i, :cur_len] = cur_new_seg_token_mask

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, seg_token_mask_padded, mask_input_dict

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

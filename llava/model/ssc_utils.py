SSC_ABLATION_MODES = ("none", "wo_sent", "wo_sframe", "wo_srel", "wo_sscene")
SSC_ROLE_ANALYSIS_VARIANTS = (
    "none",
    "generic_query_states",
    "shared_evidence_role_states",
    "object_centric_frame_evidence",
    "relation_from_primitive_evidence",
    "wo_all_role_preserving_losses",
)


def normalize_ssc_ablation_mode(mode):
    normalized = "none" if mode is None else str(mode).strip().lower()
    if normalized == "":
        normalized = "none"
    if normalized not in SSC_ABLATION_MODES:
        raise ValueError(
            "ssc_ablation_mode must be one of "
            f"{list(SSC_ABLATION_MODES)}, got {mode}."
        )
    return normalized


def normalize_ssc_role_analysis_variant(variant):
    normalized = "none" if variant is None else str(variant).strip().lower()
    if normalized == "":
        normalized = "none"
    alias_map = {
        "generic": "generic_query_states",
        "generic_queries": "generic_query_states",
        "shared_evidence": "shared_evidence_role_states",
        "shared_evidence_states": "shared_evidence_role_states",
        "object_centric_frame": "object_centric_frame_evidence",
        "relation_from_primitives": "relation_from_primitive_evidence",
        "wo_all_role_losses": "wo_all_role_preserving_losses",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized not in SSC_ROLE_ANALYSIS_VARIANTS:
        raise ValueError(
            "ssc_role_analysis_variant must be one of "
            f"{list(SSC_ROLE_ANALYSIS_VARIANTS)}, got {variant}."
        )
    return normalized


def resolve_ssc_role_analysis_settings(variant, state_total):
    normalized = normalize_ssc_role_analysis_variant(variant)
    state_total = int(state_total)
    if state_total <= 0:
        raise ValueError(f"state_total must be > 0, got {state_total}.")

    settings = {
        "variant": normalized,
        "use_generic_states": False,
        "generic_state_slots": state_total,
        "use_role_specific_token_embeddings": True,
        "use_generic_state_embedding": False,
        "entity_evidence_source": "entity_object_topk",
        "frame_evidence_source": "geometry_frame_primitives",
        "relation_source": "state_interaction",
        "frame_cov_loss_enabled": True,
        "rel_geo_loss_enabled": True,
        "ent_sem_loss_enabled": True,
        "seg_role_bias_enabled": True,
        "notes": [],
    }

    if normalized == "generic_query_states":
        settings.update(
            {
                "use_generic_states": True,
                "use_role_specific_token_embeddings": False,
                "use_generic_state_embedding": True,
                "entity_evidence_source": "disabled_generic_queries",
                "frame_evidence_source": "disabled_generic_queries",
                "relation_source": "generic_superpoint_pool",
                "frame_cov_loss_enabled": False,
                "rel_geo_loss_enabled": False,
                "ent_sem_loss_enabled": False,
                "seg_role_bias_enabled": False,
                "notes": [
                    "generic_states_cross_attend_to_superpoint_pool",
                    "role_specific_token_embeddings_disabled",
                    "role_preserving_losses_disabled",
                ],
            }
        )
    elif normalized == "shared_evidence_role_states":
        settings.update(
            {
                "entity_evidence_source": "shared_object_pool",
                "frame_evidence_source": "shared_object_pool",
                "frame_cov_loss_enabled": False,
                "notes": [
                    "entity_and_frame_states_share_object_centric_pool",
                    "frame_coverage_loss_disabled_no_geometry_groups",
                ],
            }
        )
    elif normalized == "object_centric_frame_evidence":
        settings.update(
            {
                "frame_evidence_source": "frame_object_topk",
                "frame_cov_loss_enabled": False,
                "notes": [
                    "frame_states_use_object_centric_topk_primitives",
                    "frame_coverage_loss_disabled_no_geometry_groups",
                ],
            }
        )
    elif normalized == "relation_from_primitive_evidence":
        settings.update(
            {
                "relation_source": "primitive_evidence",
                "notes": [
                    "relation_states_read_from_entity_and_frame_primitives",
                ],
            }
        )
    elif normalized == "wo_all_role_preserving_losses":
        settings.update(
            {
                "frame_cov_loss_enabled": False,
                "rel_geo_loss_enabled": False,
                "ent_sem_loss_enabled": False,
                "notes": [
                    "all_role_preserving_auxiliary_losses_disabled",
                ],
            }
        )

    return settings


def resolve_ssc_ablation_layout(
    num_pc_tokens,
    ssc_state_ent_slots,
    ssc_state_frame_slots,
    ssc_state_rel_slots,
    ssc_scene_summary_token_count,
    ssc_detail_tokens,
    ssc_ablation_mode="none",
):
    ablation_mode = normalize_ssc_ablation_mode(ssc_ablation_mode)
    num_pc_tokens = int(num_pc_tokens)
    original_ent_slots = int(ssc_state_ent_slots)
    original_frame_slots = int(ssc_state_frame_slots)
    original_rel_slots = int(ssc_state_rel_slots)
    original_scene_summary_token_count = int(ssc_scene_summary_token_count)
    if original_scene_summary_token_count < 0:
        raise ValueError(
            "ssc_scene_summary_token_count must be >= 0, "
            f"got {ssc_scene_summary_token_count}."
        )
    original_state_total = (
        original_ent_slots
        + original_frame_slots
        + original_rel_slots
        + original_scene_summary_token_count
    )
    configured_detail_tokens = int(ssc_detail_tokens)
    if configured_detail_tokens < 0:
        base_detail_tokens = num_pc_tokens - original_state_total
    else:
        base_detail_tokens = configured_detail_tokens
    if base_detail_tokens < 0:
        raise ValueError(
            "SSC3D detail token budget must be >= 0: "
            f"num_pc_tokens={num_pc_tokens}, state_total={original_state_total}."
        )

    use_ent = ablation_mode != "wo_sent"
    use_frame = ablation_mode != "wo_sframe"
    use_rel = ablation_mode != "wo_srel"
    use_scene = ablation_mode != "wo_sscene"

    deleted_token_budget = 0
    if not use_ent:
        deleted_token_budget += original_ent_slots
    if not use_frame:
        deleted_token_budget += original_frame_slots
    if not use_rel:
        deleted_token_budget += original_rel_slots
    if not use_scene:
        deleted_token_budget += original_scene_summary_token_count

    effective_ent_slots = original_ent_slots if use_ent else 0
    effective_frame_slots = original_frame_slots if use_frame else 0
    effective_rel_slots = original_rel_slots if use_rel else 0
    effective_scene_summary_token_count = original_scene_summary_token_count if use_scene else 0
    effective_state_total = (
        effective_ent_slots
        + effective_frame_slots
        + effective_rel_slots
        + effective_scene_summary_token_count
    )
    effective_detail_tokens = base_detail_tokens + deleted_token_budget
    final_token_count = effective_detail_tokens + effective_state_total

    return {
        "ablation_mode": ablation_mode,
        "use_ent": bool(use_ent),
        "use_frame": bool(use_frame),
        "use_rel": bool(use_rel),
        "use_scene": bool(use_scene),
        "original_ent_slots": original_ent_slots,
        "original_frame_slots": original_frame_slots,
        "original_rel_slots": original_rel_slots,
        "original_scene_summary_token_count": original_scene_summary_token_count,
        "effective_ent_slots": effective_ent_slots,
        "effective_frame_slots": effective_frame_slots,
        "effective_rel_slots": effective_rel_slots,
        "effective_scene_summary_token_count": effective_scene_summary_token_count,
        "original_state_total": original_state_total,
        "effective_state_total": effective_state_total,
        "base_detail_tokens": base_detail_tokens,
        "effective_detail_tokens": effective_detail_tokens,
        "deleted_token_budget": deleted_token_budget,
        "expected_total_tokens": num_pc_tokens,
        "final_token_count": final_token_count,
    }

#!/bin/bash

# SceneScaffold (NeurIPS 2026 Spotlight) - main training script.
#
# Typed superpoint primitives -> entity / scene-frame / relation / global scene states
# -> state serialization under a fixed visual-token budget -> state-driven segmentation reader.
# Internally the variant is named `ssc3d` (scene-state construction) with the `su` suffix
# denoting the default configuration (scene summary token + entity semantic retention + detail dropout).
# Every hyper-parameter below can be overridden through environment variables.

EXP_NAME=${EXP_NAME:-finetune-3d-llava-ssc3d-su-lora}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-liuhaotian/llava-v1.5-7b}
STAGE1_CKPT=${STAGE1_CKPT:-./checkpoints/pc_pretrained/ost-sa-only-llava-align-scannet200.pth}
NUM_PC_TOKENS=${NUM_PC_TOKENS:-100}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
DEEPSPEED_MASTER_PORT=${DEEPSPEED_MASTER_PORT:-29528}
OUTPUT_ROOT=${OUTPUT_ROOT:-./checkpoints}
# Optional: set SNAPSHOT_CODE=1 to archive a copy of ./llava next to the run for reproducibility.
SNAPSHOT_CODE=${SNAPSHOT_CODE:-0}
CODE_ROOT=${CODE_ROOT:-./record}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}
TRAIN_SEED=${TRAIN_SEED:-42}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}

DVC_ENABLE=${DVC_ENABLE:-True}
DVC_VARIANT=${DVC_VARIANT:-ssc3d}
DVC_SECOND_TOKEN_SEMANTICS=${DVC_SECOND_TOKEN_SEMANTICS:-frame_anchor}
# Unused in SSC3D scene-state construction; kept only for parser compatibility.
DVC_ENTITY_RATIO=${DVC_ENTITY_RATIO:-0.9}

DVC_NUM_BOUNDARY_ANCHORS=${DVC_NUM_BOUNDARY_ANCHORS:-4}
DVC_REGION_GRID_X=${DVC_REGION_GRID_X:-3}
DVC_REGION_GRID_Y=${DVC_REGION_GRID_Y:-3}
DVC_REL_BAND_RATIO=${DVC_REL_BAND_RATIO:-0.15}
DVC_REL_MIN_POINTS=${DVC_REL_MIN_POINTS:-4}

SSC_STATE_ENT_SLOTS=${SSC_STATE_ENT_SLOTS:-8}
SSC_STATE_FRAME_SLOTS=${SSC_STATE_FRAME_SLOTS:-8}
SSC_STATE_REL_SLOTS=${SSC_STATE_REL_SLOTS:-8}
SSC_SCENE_SUMMARY_TOKEN_COUNT=${SSC_SCENE_SUMMARY_TOKEN_COUNT:-1}
SSC_ENTITY_CANDIDATE_TOPK=${SSC_ENTITY_CANDIDATE_TOPK:-128}
SSC_DETAIL_TOKENS=${SSC_DETAIL_TOKENS:--1}
SSC_SEG_ALPHA_INIT=${SSC_SEG_ALPHA_INIT:-0.0}
SSC_FRAME_COV_LOSS_WEIGHT=${SSC_FRAME_COV_LOSS_WEIGHT:-0.05}
SSC_REL_BOUNDARY_LOSS_WEIGHT=${SSC_REL_BOUNDARY_LOSS_WEIGHT:-0.05}
SSC_REL_REGION_LOSS_WEIGHT=${SSC_REL_REGION_LOSS_WEIGHT:-0.05}
SSC_ENT_SEM_LOSS_WEIGHT=${SSC_ENT_SEM_LOSS_WEIGHT:-0.05}
SSC_DETAIL_DROPOUT_RATE=${SSC_DETAIL_DROPOUT_RATE:-0.2}

SSC_STATE_TOTAL=$((SSC_STATE_ENT_SLOTS + SSC_STATE_FRAME_SLOTS + SSC_STATE_REL_SLOTS + SSC_SCENE_SUMMARY_TOKEN_COUNT))
SSC_FRAME_PRIMITIVE_BUDGET=$((DVC_NUM_BOUNDARY_ANCHORS + DVC_REGION_GRID_X * DVC_REGION_GRID_Y))
if [ "${SSC_STATE_FRAME_SLOTS}" -gt "${SSC_FRAME_PRIMITIVE_BUDGET}" ]; then
    SSC_FRAME_PRIMITIVE_BUDGET=${SSC_STATE_FRAME_SLOTS}
fi
if [ "${SSC_DETAIL_TOKENS}" -lt 0 ]; then
    SSC_DETAIL_RESOLVED=$((NUM_PC_TOKENS - SSC_STATE_TOTAL))
else
    SSC_DETAIL_RESOLVED=${SSC_DETAIL_TOKENS}
fi
if [ "${SSC_DETAIL_RESOLVED}" -lt 0 ]; then
    echo "ERROR: resolved residual token count is negative: ${SSC_DETAIL_RESOLVED}" >&2
    exit 2
fi
if [ $((SSC_DETAIL_RESOLVED + SSC_STATE_TOTAL)) -ne "${NUM_PC_TOKENS}" ]; then
    echo "ERROR: token allocation does not sum to NUM_PC_TOKENS" >&2
    exit 2
fi

# referring seg
export scanrefer=./playground/data/train_info/scanrefer_train_3d_llava.json
export multi3drefer=./playground/data/train_info/multi3drefer_train_3d_llava.json
export nr3d=./playground/data/train_info/nr3d_train_3d_llava.json

# dense captioning
export scan2cap=./playground/data/train_info/scan2cap_train_3d_llava.json
export nr3d_caption=./playground/data/train_info/nr3d_caption_train_3d_llava.json

# vqa
export scanqa=./playground/data/train_info/scanqa_train_3d_llava.json
export sqa3d=./playground/data/train_info/sqa3d_train_3d_llava.json

echo "EXP_NAME=${EXP_NAME}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "STAGE1_CKPT=${STAGE1_CKPT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "CODE_ROOT=${CODE_ROOT}"
echo "DVC_ENABLE=${DVC_ENABLE}"
echo "DVC_VARIANT=${DVC_VARIANT}"
echo "DVC_SECOND_TOKEN_SEMANTICS=${DVC_SECOND_TOKEN_SEMANTICS}"
echo "NUM_PC_TOKENS=${NUM_PC_TOKENS}"
echo "DVC_NUM_BOUNDARY_ANCHORS=${DVC_NUM_BOUNDARY_ANCHORS}"
echo "DVC_REGION_GRID_X=${DVC_REGION_GRID_X}"
echo "DVC_REGION_GRID_Y=${DVC_REGION_GRID_Y}"
echo "DVC_REL_BAND_RATIO=${DVC_REL_BAND_RATIO}"
echo "DVC_REL_MIN_POINTS=${DVC_REL_MIN_POINTS}"
echo "SSC_STATE_TOTAL=${SSC_STATE_TOTAL}"
echo "SSC_SCENE_SUMMARY_TOKEN_COUNT=${SSC_SCENE_SUMMARY_TOKEN_COUNT}"
echo "SSC_FRAME_PRIMITIVE_BUDGET=${SSC_FRAME_PRIMITIVE_BUDGET}"
echo "SSC_STATE_ENT_SLOTS=${SSC_STATE_ENT_SLOTS}"
echo "SSC_STATE_FRAME_SLOTS=${SSC_STATE_FRAME_SLOTS}"
echo "SSC_STATE_REL_SLOTS=${SSC_STATE_REL_SLOTS}"
echo "SSC_SCENE_SUMMARY_TOKEN_COUNT=${SSC_SCENE_SUMMARY_TOKEN_COUNT}"
echo "SSC_ENTITY_CANDIDATE_TOPK=${SSC_ENTITY_CANDIDATE_TOPK}"
echo "SSC_DETAIL_TOKENS=${SSC_DETAIL_TOKENS}"
echo "SSC_DETAIL_RESOLVED=${SSC_DETAIL_RESOLVED}"
echo "SSC_SEG_ALPHA_INIT=${SSC_SEG_ALPHA_INIT}"
echo "SSC_FRAME_COV_LOSS_WEIGHT=${SSC_FRAME_COV_LOSS_WEIGHT}"
echo "SSC_REL_BOUNDARY_LOSS_WEIGHT=${SSC_REL_BOUNDARY_LOSS_WEIGHT}"
echo "SSC_REL_REGION_LOSS_WEIGHT=${SSC_REL_REGION_LOSS_WEIGHT}"
echo "SSC_ENT_SEM_LOSS_WEIGHT=${SSC_ENT_SEM_LOSS_WEIGHT}"
echo "SSC_DETAIL_DROPOUT_RATE=${SSC_DETAIL_DROPOUT_RATE}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "DEEPSPEED_MASTER_PORT=${DEEPSPEED_MASTER_PORT}"
echo "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
echo "TRAIN_SEED=${TRAIN_SEED}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"

DEEPSPEED_LAUNCH_ARGS=()
TRAIN_OPTIONAL_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    TRAIN_OPTIONAL_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    SSC_ORIG_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES// /}"
    SSC_DEEPSPEED_INCLUDE="localhost:${SSC_ORIG_CUDA_VISIBLE_DEVICES}"
    DEEPSPEED_LAUNCH_ARGS+=(--include "${SSC_DEEPSPEED_INCLUDE}")
    echo "CUDA_VISIBLE_DEVICES=${SSC_ORIG_CUDA_VISIBLE_DEVICES}"
    echo "SSC_DEEPSPEED_INCLUDE=${SSC_DEEPSPEED_INCLUDE}"
    echo "SSC_DEEPSPEED_GPU_MODE=physical_slots"
    unset CUDA_VISIBLE_DEVICES
fi

if [ "${SNAPSHOT_CODE}" = "1" ]; then
    CODE_DIR=${CODE_ROOT}/${EXP_NAME}
    mkdir -p "$CODE_DIR"
    cp -r llava "$CODE_DIR"
fi

PYTHONPATH=$(pwd) \
deepspeed "${DEEPSPEED_LAUNCH_ARGS[@]}" --master_port "${DEEPSPEED_MASTER_PORT}" llava/train/train_mem.py \
    --lora_enable True --lora_r 32 --lora_alpha 64 \
    --deepspeed ./scripts/zero1_3d_llava.json \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version v1 \
    --data_path $scan2cap $scanqa $sqa3d $nr3d_caption $scanrefer $scanrefer $scanrefer $multi3drefer $nr3d \
    --scan_folder ./playground/data/scannet \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --pointcloud_tower "${STAGE1_CKPT}" \
    --pc_modules_to_finetune alignment_proj hidden_seg_fc \
    --num_pc_tokens ${NUM_PC_TOKENS} \
    --dvc_enable ${DVC_ENABLE} \
    --dvc_phase_schedule none \
    --dvc_variant ${DVC_VARIANT} \
    --dvc_entity_ratio ${DVC_ENTITY_RATIO} \
    --dvc_second_token_semantics ${DVC_SECOND_TOKEN_SEMANTICS} \
    --dvc_num_boundary_anchors ${DVC_NUM_BOUNDARY_ANCHORS} \
    --dvc_region_grid_x ${DVC_REGION_GRID_X} \
    --dvc_region_grid_y ${DVC_REGION_GRID_Y} \
    --dvc_rel_band_ratio ${DVC_REL_BAND_RATIO} \
    --dvc_rel_min_points ${DVC_REL_MIN_POINTS} \
    --ssc_state_ent_slots ${SSC_STATE_ENT_SLOTS} \
    --ssc_state_frame_slots ${SSC_STATE_FRAME_SLOTS} \
    --ssc_state_rel_slots ${SSC_STATE_REL_SLOTS} \
    --ssc_scene_summary_token_count ${SSC_SCENE_SUMMARY_TOKEN_COUNT} \
    --ssc_entity_candidate_topk ${SSC_ENTITY_CANDIDATE_TOPK} \
    --ssc_detail_tokens ${SSC_DETAIL_TOKENS} \
    --ssc_seg_alpha_init ${SSC_SEG_ALPHA_INIT} \
    --ssc_frame_cov_loss_weight ${SSC_FRAME_COV_LOSS_WEIGHT} \
    --ssc_rel_boundary_loss_weight ${SSC_REL_BOUNDARY_LOSS_WEIGHT} \
    --ssc_rel_region_loss_weight ${SSC_REL_REGION_LOSS_WEIGHT} \
    --ssc_ent_sem_loss_weight ${SSC_ENT_SEM_LOSS_WEIGHT} \
    --ssc_detail_dropout_rate ${SSC_DETAIL_DROPOUT_RATE} \
    "${TRAIN_OPTIONAL_ARGS[@]}" \
    --inst_prompt_encoder shared_projector \
    --freeze_pointcloud_tower True \
    --pc_use_link_token False \
    --image_aspect_ratio pad \
    --group_by_task_length_per_batch True \
    --bf16 True \
    --output_dir "${OUTPUT_ROOT}/${EXP_NAME}" \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --seed ${TRAIN_SEED} \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH=$(pwd)

GROUNDING_ROOT=${GROUNDING_ROOT:-./outputs/grounding_metrics}
PRED_ROOT=${PRED_ROOT:-$GROUNDING_ROOT/scanrefer}
BOX_DBSCAN_EPS=${BOX_DBSCAN_EPS:-0.08}
BOX_DBSCAN_MIN_SAMPLES=${BOX_DBSCAN_MIN_SAMPLES:-10}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

EXP_NAME=${EXP_NAME:-finetune-3d-llava-ssc3d-su-lora}
CKPT_PATH=${CKPT_PATH:-./checkpoints/finetune-3d-llava-ssc3d-su-lora}
MODEL_BASE=${MODEL_BASE:-liuhaotian/llava-v1.5-7b}

echo "EXP_NAME=${EXP_NAME}"
echo "CKPT_PATH=${CKPT_PATH}"
echo "MODEL_BASE=${MODEL_BASE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [ ! -d "$CKPT_PATH" ]; then
    echo "[ERROR] checkpoint path not found: $CKPT_PATH"
    exit 1
fi

if [[ "$EXP_NAME" != *lora* && "$CKPT_PATH" != *lora* ]]; then
    echo "[WARNING] model path/name does not contain 'lora'; please ensure builder uses the intended LoRA loading branch."
fi

mkdir -p "$PRED_ROOT"
if [[ "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        [[ ! -e "$PRED_ROOT/${CHUNKS}_${IDX}.jsonl" ]] || { echo "[ERROR] chunk exists; refusing to overwrite: $PRED_ROOT/${CHUNKS}_${IDX}.jsonl" >&2; exit 1; }
    done
    [[ ! -e "$PRED_ROOT/merge.jsonl" ]] || { echo "[ERROR] output exists; refusing to overwrite: $PRED_ROOT/merge.jsonl" >&2; exit 1; }
    [[ ! -e "$PRED_ROOT/box_metrics.json" ]] || { echo "[ERROR] metrics exists; refusing to overwrite: $PRED_ROOT/box_metrics.json" >&2; exit 1; }
fi

pids=()

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_scanrefer \
        --scan-folder ./playground/data/scannet/val \
        --model-path "$CKPT_PATH" \
        --model-base $MODEL_BASE \
        --question-file ./playground/data/eval_info/referseg_scanrefer/ScanRefer_filtered_val.json \
        --answers-file "$PRED_ROOT/${CHUNKS}_${IDX}.jsonl" \
        --box-dbscan-eps "$BOX_DBSCAN_EPS" \
        --box-dbscan-min-samples "$BOX_DBSCAN_MIN_SAMPLES" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --conv-mode vicuna_v1 &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

output_file="$PRED_ROOT/merge.jsonl"

if [[ -e "$output_file" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
    echo "[ERROR] output exists; refusing to overwrite: $output_file" >&2
    exit 1
fi
: > "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    chunk_file="$PRED_ROOT/${CHUNKS}_${IDX}.jsonl"
    if [ ! -f "$chunk_file" ]; then
        echo "[ERROR] missing chunk prediction file: $chunk_file"
        exit 1
    fi
    cat "$chunk_file" >> "$output_file"
done

python llava/eval/eval_refer_seg.py \
    --result-file "$output_file"

python llava/eval/eval_scanrefer_box.py \
    --result-file "$output_file" \
    --question-file ./playground/data/eval_info/referseg_scanrefer/ScanRefer_filtered_val.json \
    --scan-folder ./playground/data/scannet/val \
    --output-json "$PRED_ROOT/box_metrics.json"

#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTHONPATH=$(pwd)

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

EXP_NAME=${EXP_NAME:-finetune-3d-llava-ssc3d-su-lora}
CKPT_PATH=${CKPT_PATH:-./checkpoints/${EXP_NAME}}
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

mkdir -p "./playground/predictions/$EXP_NAME/sqa3d"

pids=()

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_sqa3d \
        --scan-folder ./playground/data/scannet/val \
        --model-path $CKPT_PATH \
        --model-base $MODEL_BASE \
        --question-file ./playground/data/eval_info/sqa3d/sqa3d_test_question.json \
        --answers-file ./playground/predictions/$EXP_NAME/sqa3d/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

output_file=./playground/predictions/$EXP_NAME/sqa3d/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    chunk_file=./playground/predictions/$EXP_NAME/sqa3d/${CHUNKS}_${IDX}.jsonl
    if [ ! -f "$chunk_file" ]; then
        echo "[ERROR] missing chunk prediction file: $chunk_file"
        exit 1
    fi
    cat "$chunk_file" >> "$output_file"
done

python llava/eval/eval_sqa3d.py \
    --annotation-file ./playground/data/eval_info/sqa3d/sqa3d_test_answer.json \
    --result-file $output_file

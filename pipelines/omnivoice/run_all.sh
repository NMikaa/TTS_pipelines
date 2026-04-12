#!/bin/bash
# OmniVoice Georgian fine-tuning — 3 quality thresholds
# Run: bash pipelines/omnivoice/run_all.sh [stage] [stop_stage] [subset]
# Examples:
#   bash run_all.sh 0 1          # tokenize + train all 3
#   bash run_all.sh 0 0          # tokenize only
#   bash run_all.sh 1 1 095      # train only 0.95 subset
#   bash run_all.sh 1 1 097      # train only 0.97 subset

set -euo pipefail

# Disable torch.compile (Triton shared memory issues on some GPUs)
export TORCH_COMPILE_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../OmniVoice" && pwd)"
# Set VENV to your OmniVoice venv (default: ../../venv_omnivoice/bin)
VENV="${VENV:-$(cd "${SCRIPT_DIR}/../../venv_omnivoice/bin" 2>/dev/null && pwd)}"

stage=${1:-0}
stop_stage=${2:-1}
subset=${3:-"all"}  # "all", "095", "097", "099"

GPU_IDS="0,1"
NUM_GPUS=2
TOKENIZER="eustlb/higgs-audio-v2-tokenizer"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

SUBSETS=("095" "097" "099")
if [ "$subset" != "all" ]; then
    SUBSETS=("$subset")
fi

# ============================================================================
# Stage 0: Tokenize audio into WebDataset shards
# ============================================================================
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    for tag in "${SUBSETS[@]}"; do
        DATA_DIR="${SCRIPT_DIR}/data/${tag}"
        TOKEN_DIR="${SCRIPT_DIR}/data/${tag}/tokens"

        for split in train dev; do
            JSONL="${DATA_DIR}/${split}.jsonl"
            if [ ! -f "$JSONL" ]; then
                echo "Skipping ${tag}/${split} — no JSONL"
                continue
            fi

            if [ -f "${TOKEN_DIR}/${split}/data.lst" ]; then
                echo "Skipping ${tag}/${split} — already tokenized"
                continue
            fi

            echo "=== Tokenizing ${tag}/${split} ==="
            CUDA_VISIBLE_DEVICES=${GPU_IDS} \
                ${VENV}/python -m omnivoice.scripts.extract_audio_tokens \
                --input_jsonl "${JSONL}" \
                --tar_output_pattern "${TOKEN_DIR}/${split}/audios/shard-%06d.tar" \
                --jsonl_output_pattern "${TOKEN_DIR}/${split}/txts/shard-%06d.jsonl" \
                --tokenizer_path "${TOKENIZER}" \
                --nj_per_gpu 3 \
                --shuffle True

            echo "Done: ${TOKEN_DIR}/${split}/data.lst"
        done
    done
fi

# ============================================================================
# Stage 1: Fine-tune (sequential — one subset at a time)
# ============================================================================
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    TRAIN_CONFIG="${SCRIPT_DIR}/config/train_config.json"

    for tag in "${SUBSETS[@]}"; do
        TOKEN_DIR="${SCRIPT_DIR}/data/${tag}/tokens"
        OUTPUT_DIR="${SCRIPT_DIR}/exp/${tag}"
        DATA_CONFIG="${SCRIPT_DIR}/config/data_config_${tag}.json"

        # Generate data config pointing to tokenized shards
        cat > "${DATA_CONFIG}" << DCEOF
{
    "train": [
        {"manifest_path": ["${TOKEN_DIR}/train/data.lst"]}
    ],
    "dev": [
        {"manifest_path": ["${TOKEN_DIR}/dev/data.lst"]}
    ]
}
DCEOF

        echo "=== Training ${tag} ==="
        echo "  Data: ${TOKEN_DIR}"
        echo "  Output: ${OUTPUT_DIR}"

        ${VENV}/accelerate launch \
            --gpu_ids "${GPU_IDS}" \
            --num_processes ${NUM_GPUS} \
            -m omnivoice.cli.train \
            --train_config "${TRAIN_CONFIG}" \
            --data_config "${DATA_CONFIG}" \
            --output_dir "${OUTPUT_DIR}"

        echo "=== Done training ${tag} ==="
    done
fi

echo "All done!"

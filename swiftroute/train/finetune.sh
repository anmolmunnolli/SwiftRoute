#!/usr/bin/env bash
# Phase 6b: distil qwen2.5:7b's behaviour into Qwen2.5-1.5B with LoRA, on an Apple-silicon Mac.
# Free: runs locally with MLX. Rough time on an M-series Air: data ~1 h, training ~20-40 min.
#
#   bash train/finetune.sh              # full run
#   STUDENT=Qwen/Qwen2.5-3B-Instruct bash train/finetune.sh
set -euo pipefail
cd "$(dirname "$0")/.."

STUDENT="${STUDENT:-Qwen/Qwen2.5-1.5B-Instruct}"
TEACHER="${TEACHER:-qwen2.5:7b}"
VARIANTS="${VARIANTS:-8}"
ITERS="${ITERS:-600}"
ADAPTER="data/adapters/$(basename "$STUDENT")-swiftroute"

pip install -q "mlx-lm>=0.32"

# 1. teacher conversations -> data/sft/{train,valid}.jsonl  (skip if already built)
if [ ! -s data/sft/train.jsonl ]; then
  python train/make_dataset.py --teacher "$TEACHER" --variants "$VARIANTS"
fi
echo "train examples: $(wc -l < data/sft/train.jsonl)  valid: $(wc -l < data/sft/valid.jsonl)"

# 2. LoRA. --mask-prompt: loss only on the reply, not on the (long) prompt.
#    --grad-checkpoint + batch 2: fits an 8-16 GB Air at ~1.5-2k-token prompts.
mlx_lm.lora \
  --model "$STUDENT" \
  --train \
  --data data/sft \
  --mask-prompt \
  --num-layers 16 \
  --batch-size 2 \
  --grad-accumulation-steps 4 \
  --iters "$ITERS" \
  --learning-rate 1e-4 \
  --max-seq-length 4096 \
  --grad-checkpoint \
  --steps-per-eval 100 \
  --save-every 200 \
  --adapter-path "$ADAPTER"

echo
echo "adapter saved to $ADAPTER"
echo "serve it:    bash train/serve.sh"
echo "compare it:  python evals/compare.py"

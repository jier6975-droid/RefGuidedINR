#!/usr/bin/env bash
# run_eval.sh — Launch RefGuidedINR evaluation.
# Usage: bash scripts/run_eval.sh [config] [checkpoint] [gpu]

set -euo pipefail

CONFIG="${1:-configs/example_config.yaml}"
CHECKPOINT="${2:-checkpoints/best.pth}"
GPU="${3:-0}"

# Move to repo root
cd "$(dirname "$0")/.."

echo "========================================"
echo "  RefGuidedINR Evaluation"
echo "  Config     : $CONFIG"
echo "  Checkpoint : $CHECKPOINT"
echo "  GPU        : $GPU"
echo "========================================"

python evaluators/evaluate.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --gpu "$GPU"

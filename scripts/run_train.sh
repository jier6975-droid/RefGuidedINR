#!/usr/bin/env bash
# run_train.sh — Launch RefGuidedINR training.
# Usage: bash scripts/run_train.sh [config] [gpu]

set -euo pipefail

CONFIG="${1:-configs/example_config.yaml}"
GPU="${2:-0}"

# Move to repo root
cd "$(dirname "$0")/.."

echo "========================================"
echo "  RefGuidedINR Training"
echo "  Config : $CONFIG"
echo "  GPU    : $GPU"
echo "========================================"

python trainers/train.py --config "$CONFIG" --gpu "$GPU"

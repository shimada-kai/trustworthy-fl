#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Centralized Learning Baseline ==="

python -m secure_fl.experiments.centralized \
  --epochs 5 \
  --batch-size 256 \
  --lr 0.001

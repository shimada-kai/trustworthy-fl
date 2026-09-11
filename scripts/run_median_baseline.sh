#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Coordinate-wise Median Baseline ==="

flwr run . \
  --run-config 'aggregation-type="median" attack-enabled=false tee-enabled=false zk-verification-enabled=false num-server-rounds=5' \
  --stream

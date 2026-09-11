#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== FedAvg Baseline ==="

flwr run . \
  --run-config 'aggregation-type="fedavg" attack-enabled=false tee-enabled=false zk-verification-enabled=false num-server-rounds=5' \
  --stream

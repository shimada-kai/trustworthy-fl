#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== FedAvg + Sign Flip (3/7 attackers) ==="

flwr run . \
  --run-config 'aggregation-type="fedavg" attack-enabled=true attack-type="sign_flip" malicious-client-ids="0,1,2" attack-scale=1.0 tee-enabled=false zk-verification-enabled=false num-server-rounds=5' \
  --stream

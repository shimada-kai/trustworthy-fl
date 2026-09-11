#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source scripts/lib/tdx_vm.sh

cleanup() {
  local exit_code=$?

  trap - EXIT INT TERM

  if ! stop_tdx_vm_if_started; then
    echo "🚨 Automatic TDX VM shutdown verification failed."
    exit 1
  fi

  exit "$exit_code"
}

trap cleanup EXIT INT TERM

echo "=== ZK Filter + Intel TDX Median + Sign Flip (4/7 attackers) ==="

start_tdx_vm

echo
echo "Starting federated learning experiment..."
echo

flwr run . \
  --run-config 'aggregation-type="median" attack-enabled=true attack-type="sign_flip" malicious-client-ids="0,1,2,3" attack-scale=1.0 tee-enabled=true zk-verification-enabled=true num-server-rounds=5' \
  --stream

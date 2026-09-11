#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "========================================"
echo " Trustworthy FL - Local Experiments"
echo "========================================"
echo
echo "TDX experiment is intentionally excluded."
echo "Run scripts/run_tdx_zk_median_sign_flip.sh separately."
echo

./scripts/run_centralized.sh
./scripts/run_fedavg_baseline.sh
./scripts/run_fedavg_sign_flip.sh
./scripts/run_median_baseline.sh
./scripts/run_median_sign_flip.sh
./scripts/run_median_sign_flip_4of7.sh
./scripts/run_zk_median_sign_flip.sh

echo
echo "========================================"
echo " All local experiments completed"
echo "========================================"

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

echo "=== Trustworthy FL Streamlit Demo ==="

start_tdx_vm

echo
echo "Starting Streamlit demo..."
echo "When finished, press Ctrl+C in this terminal."
echo "The TDX VM will then be stopped automatically."
echo

streamlit run demo/app.py

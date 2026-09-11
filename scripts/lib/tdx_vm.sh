#!/usr/bin/env bash

TDX_VM_NAME="${TDX_VM_NAME:-tdx-demo}"
TDX_VM_ZONE="${TDX_VM_ZONE:-asia-northeast1-b}"
TDX_VM_STARTED_BY_SCRIPT=false

tdx_vm_status() {
  gcloud compute instances describe "$TDX_VM_NAME" \
    --zone="$TDX_VM_ZONE" \
    --format='get(status)'
}

wait_for_tdx_status() {
  local expected_status="$1"
  local max_attempts="${2:-60}"

  for ((attempt=1; attempt<=max_attempts; attempt++)); do
    local status
    status="$(tdx_vm_status 2>/dev/null || true)"

    if [[ "$status" == "$expected_status" ]]; then
      return 0
    fi

    sleep 5
  done

  return 1
}

start_tdx_vm() {
  local status
  status="$(tdx_vm_status)"

  if [[ "$status" == "RUNNING" ]]; then
    echo "TDX VM is already RUNNING."
    echo "This script did not start it, so it will not stop it automatically."
    TDX_VM_STARTED_BY_SCRIPT=false
    return 0
  fi

  echo
  echo "========================================================"
  echo "⚠️  GCP BILLING WARNING"
  echo "Starting Intel TDX VM: $TDX_VM_NAME"
  echo "Compute charges begin when the VM starts running."
  echo "========================================================"
  echo

  gcloud compute instances start "$TDX_VM_NAME" \
    --zone="$TDX_VM_ZONE"

  if ! wait_for_tdx_status "RUNNING"; then
    echo "ERROR: TDX VM did not reach RUNNING state."
    return 1
  fi

  TDX_VM_STARTED_BY_SCRIPT=true

  echo "✅ TDX VM START CONFIRMED"
  echo "Status: $(tdx_vm_status)"
}

stop_tdx_vm_if_started() {
  if [[ "$TDX_VM_STARTED_BY_SCRIPT" != "true" ]]; then
    echo "TDX VM was not started by this script; automatic stop skipped."
    return 0
  fi

  echo
  echo "Stopping TDX VM..."
  gcloud compute instances stop "$TDX_VM_NAME" \
    --zone="$TDX_VM_ZONE"

  if wait_for_tdx_status "TERMINATED"; then
    echo
    echo "========================================================"
    echo "✅ TDX VM STOP CONFIRMED"
    echo "VM: $TDX_VM_NAME"
    echo "Zone: $TDX_VM_ZONE"
    echo "Status: TERMINATED"
    echo "Compute billing for the VM has stopped."
    echo "========================================================"
    return 0
  fi

  local status
  status="$(tdx_vm_status 2>/dev/null || echo UNKNOWN)"

  echo
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "�� WARNING: TDX VM STOP NOT CONFIRMED"
  echo "VM: $TDX_VM_NAME"
  echo "Current status: $status"
  echo "CHECK GOOGLE CLOUD IMMEDIATELY."
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

  return 1
}

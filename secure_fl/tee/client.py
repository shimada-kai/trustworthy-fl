import base64
import hashlib
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import requests
import torch


DEFAULT_INSTANCE = (
    "projects/26136170518/zones/asia-northeast1-b/"
    "instances/744917073552411453"
)


def _tls_spki_hash(cert_path: str) -> bytes:
    """Compute SHA-256 of the TLS certificate SubjectPublicKeyInfo."""

    public_key_pem = subprocess.check_output(
        [
            "openssl",
            "x509",
            "-in",
            cert_path,
            "-pubkey",
            "-noout",
        ]
    )

    public_key_der = subprocess.check_output(
        [
            "openssl",
            "pkey",
            "-pubin",
            "-outform",
            "DER",
        ],
        input=public_key_pem,
    )

    return hashlib.sha256(public_key_der).digest()


def _verify_remote_attestation(
    api_url: str,
    ca_cert: str,
    instance: str,
    verifier_bin: str,
    tdx_check_bin: str = "~/bin/tdx-check",
) -> None:
    """Verify that the TLS endpoint is backed by the expected TDX VM."""

    # Fresh challenge prevents replay of an old quote.
    nonce = os.urandom(32)

    response = requests.post(
        f"{api_url}/attest",
        json={"nonce": nonce.hex()},
        verify=ca_cert,
        timeout=60,
    )
    response.raise_for_status()

    evidence = response.json()

    quote = base64.b64decode(evidence["quote"], validate=True)
    server_tls_hash = bytes.fromhex(evidence["tls_key_hash"])

    # Compute the TLS key hash independently from the pinned certificate.
    expected_tls_hash = _tls_spki_hash(ca_cert)

    if server_tls_hash != expected_tls_hash:
        raise RuntimeError(
            "Remote attestation failed: TLS public-key hash mismatch"
        )

    # REPORT_DATA = fresh nonce || TLS public-key hash
    report_data = nonce + expected_tls_hash

    with tempfile.TemporaryDirectory() as temp_dir:
        quote_path = Path(temp_dir) / "quote.bin"
        quote_path.write_bytes(quote)

        # 1. Verify GCE provenance and expected VM identity.
        provenance_result = subprocess.run(
            [
                os.path.expanduser(verifier_bin),
                "verify",
                "-quote",
                str(quote_path),
                "-challenge",
                report_data.hex(),
                "-instance",
                instance,
            ],
            text=True,
            capture_output=True,
            timeout=60,
            cwd=temp_dir,
        )

        if provenance_result.returncode != 0:
            raise RuntimeError(
                "GCE provenance verification failed:\n"
                f"{provenance_result.stdout}\n"
                f"{provenance_result.stderr}"
            )

        # 2. Verify TDX TCB status, Intel collateral, CRL,
        #    and REPORT_DATA binding.
        tdx_check_result = subprocess.run(
            [
                os.path.expanduser(tdx_check_bin),
                "-in",
                str(quote_path),
                "-inform",
                "bin",
                "-report_data",
                report_data.hex(),
                "-get_collateral",
                "true",
                "-check_crl",
                "true",
                "-quiet",
            ],
            text=True,
            capture_output=True,
            timeout=180,
            cwd=temp_dir,
        )

        if tdx_check_result.returncode != 0:
            raise RuntimeError(
                "TDX TCB/CRL verification failed:\n"
                f"{tdx_check_result.stdout}\n"
                f"{tdx_check_result.stderr}"
            )

    print("[REMOTE-ATTESTATION] result=VERIFIED")
    print("[TLS-BINDING] result=VERIFIED")

def aggregate_via_tee_api(
    client_states: Sequence[Mapping[str, torch.Tensor]],
    api_url: str = "https://localhost:8000",
    ca_cert: str = "./server.crt",
    instance: str = DEFAULT_INSTANCE,
    verifier_bin: str = "~/bin/gceprovenance",
) -> dict[str, torch.Tensor]:
    """Attest the TDX VM, then send client models for median aggregation."""

    # Security gate:
    # no model parameters are sent before remote attestation succeeds.
    _verify_remote_attestation(
        api_url=api_url,
        ca_cert=ca_cert,
        instance=instance,
        verifier_bin=verifier_bin,
    )

    payload: dict[str, Any] = {
        "client_states": [
            {
                name: tensor.detach().cpu().tolist()
                for name, tensor in state.items()
            }
            for state in client_states
        ]
    }

    response = requests.post(
        f"{api_url}/aggregate",
        json=payload,
        verify=ca_cert,
        timeout=60,
    )
    response.raise_for_status()

    aggregated_state = response.json()["aggregated_state"]

    return {
        name: torch.tensor(value)
        for name, value in aggregated_state.items()
    }

def submit_update_via_tee_api(
    round_id: int,
    client_id: str,
    state: Mapping[str, torch.Tensor],
    api_url: str = "https://34.146.228.189:8000",
    ca_cert: str = "certs/server.crt",
    instance: str = DEFAULT_INSTANCE,
    verifier_bin: str = "~/bin/gceprovenance",
) -> dict[str, Any]:
    """Attest the TDX VM, then send one client update directly to it."""

    # Security gate:
    # no individual model update is sent before attestation succeeds.
    _verify_remote_attestation(
        api_url=api_url,
        ca_cert=ca_cert,
        instance=instance,
        verifier_bin=verifier_bin,
    )

    payload = {
        "round_id": round_id,
        "client_id": client_id,
        "state": {
            name: tensor.detach().cpu().tolist()
            for name, tensor in state.items()
        },
    }

    response = requests.post(
        f"{api_url}/submit_update",
        json=payload,
        verify=ca_cert,
        timeout=60,
    )
    response.raise_for_status()

    print(
        f"[TLS-SUBMIT] "
        f"round={round_id} "
        f"client={client_id} "
        f"result=SUCCESS"
    )

    return response.json()

def get_aggregated_update_via_tee_api(
    round_id: int,
    api_url: str = "https://34.146.228.189:8000",
    ca_cert: str = "certs/server.crt",
) -> tuple[dict[str, torch.Tensor], int]:
    """Fetch only the aggregated model for one round from the TDX VM."""

    response = requests.post(
        f"{api_url}/aggregate_round",
        json={"round_id": round_id},
        verify=ca_cert,
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()

    aggregated_state = {
        name: torch.tensor(value)
        for name, value in result["aggregated_state"].items()
    }

    return aggregated_state, int(result["accepted_clients"])
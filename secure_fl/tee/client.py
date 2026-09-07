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

    # Never trust the TLS hash returned by the server alone.
    # Compute it independently from the certificate pinned by this client.
    expected_tls_hash = _tls_spki_hash(ca_cert)

    if server_tls_hash != expected_tls_hash:
        raise RuntimeError(
            "Remote attestation failed: TLS public-key hash mismatch"
        )

    report_data = nonce + expected_tls_hash

    with tempfile.TemporaryDirectory() as temp_dir:
        quote_path = Path(temp_dir) / "quote.bin"
        quote_path.write_bytes(quote)

        result = subprocess.run(
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
        )

    if result.returncode != 0:
        raise RuntimeError(
            "Remote attestation failed:\n"
            f"{result.stdout}\n{result.stderr}"
        )


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
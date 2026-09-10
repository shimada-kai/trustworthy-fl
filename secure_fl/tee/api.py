import base64
import tempfile
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from secure_fl.tee.aggregator import aggregate_in_tee


app = FastAPI(title="TDX Aggregator API")

TLS_CERT_PATH = Path("certs/server.crt")
TDX_ATTEST_BIN = Path("/usr/local/sbin/tdx-attest")

# ==========================================
# Per-round verified update storage
# ==========================================
#
# Client updates are kept only inside the TDX process.
# Flower Server must never receive these individual updates.
#
# zkVM integration:
# For now, an update reaching this storage is treated as accepted.
# Later, only updates that pass zkVM verification will be stored here.
#
# {
#     round_id: {
#         client_id: state_dict,
#     }
# }
#
verified_updates: dict[int, dict[str, dict[str, Any]]] = {}

class AggregateRequest(BaseModel):
    client_states: list[dict[str, Any]]

class SubmitUpdateRequest(BaseModel):
    round_id: int
    client_id: str
    state: dict[str, Any]

class AttestRequest(BaseModel):
    nonce: str

class AggregateRoundRequest(BaseModel):
    round_id: int

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/attest")
def attest(request: AttestRequest) -> dict[str, str]:
    try:
        nonce = bytes.fromhex(request.nonce)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="nonce must be valid hexadecimal",
        ) from exc

    if len(nonce) != 32:
        raise HTTPException(
            status_code=400,
            detail="nonce must be exactly 32 bytes",
        )

    public_key_der = subprocess.check_output(
        [
            "openssl",
            "x509",
            "-in",
            str(TLS_CERT_PATH),
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
        input=public_key_der,
    )

    tls_key_hash = hashlib.sha256(public_key_der).digest()

    report_data = nonce + tls_key_hash

    with tempfile.TemporaryDirectory() as temp_dir:
        quote_path = Path(temp_dir) / "quote.bin"

        subprocess.run(
            [
                "sudo",
                "-n",
                str(TDX_ATTEST_BIN),
                "-in",
                report_data.hex(),
                "-inform",
                "hex",
                "-out",
                str(quote_path),
                "-outform",
                "bin",
            ],
            check=True,
            timeout=30,
        )

        quote = quote_path.read_bytes()

    return {
        "quote": base64.b64encode(quote).decode(),
        "tls_key_hash": tls_key_hash.hex(),
    }

@app.post("/submit_update")
def submit_update(request: SubmitUpdateRequest) -> dict[str, Any]:
    round_updates = verified_updates.setdefault(
        request.round_id,
        {},
    )

    if request.client_id in round_updates:
        raise HTTPException(
            status_code=409,
            detail=(
                f"duplicate update: "
                f"round={request.round_id}, "
                f"client={request.client_id}"
            ),
        )

    round_updates[request.client_id] = request.state

    return {
        "status": "accepted",
        "round_id": request.round_id,
        "client_id": request.client_id,
        "accepted_updates": len(round_updates),
    }

@app.post("/aggregate_round")
def aggregate_round(
    request: AggregateRoundRequest,
) -> dict[str, Any]:
    round_updates = verified_updates.get(
        request.round_id,
        {}
    )

    if not round_updates:
        raise HTTPException(
            status_code=409,
            detail=(
                f"no accepted updates for "
                f"round={request.round_id}"
            ),
        )

    aggregated_state = aggregate_in_tee(
        list(round_updates.values())
    )

    accepted_clients = len(round_updates)

    # Individual client updates are no longer needed after aggregation.
    del verified_updates[request.round_id]

    return {
        "round_id": request.round_id,
        "accepted_clients": accepted_clients,
        "aggregated_state": aggregated_state,
    }

@app.post("/aggregate")
def aggregate(request: AggregateRequest) -> dict[str, Any]:
    aggregated_state = aggregate_in_tee(request.client_states)

    return {"aggregated_state": aggregated_state}
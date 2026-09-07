import base64
import tempfile
import hashlib
import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from secure_fl.tee.aggregator import aggregate_in_tee


app = FastAPI(title="TDX Aggregator API")

TLS_CERT_PATH = Path("certs/server.crt")
TDX_ATTEST_BIN = Path("/home/shimadakai/bin/tdx-attest")


class AggregateRequest(BaseModel):
    client_states: list[dict[str, Any]]


class AttestRequest(BaseModel):
    nonce: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/attest")
def attest(request: AttestRequest) -> dict[str, str]:
    nonce = bytes.fromhex(request.nonce)

    if len(nonce) != 32:
        raise ValueError("nonce must be exactly 32 bytes")

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


@app.post("/aggregate")
def aggregate(request: AggregateRequest) -> dict[str, Any]:
    aggregated_state = aggregate_in_tee(request.client_states)

    return {"aggregated_state": aggregated_state}
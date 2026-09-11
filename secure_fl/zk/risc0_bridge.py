"""Bridge between Flower training traces and RISC Zero."""

import json
import subprocess
from pathlib import Path
from typing import Any


def export_adam_trace(
    training_trace: list[dict[str, Any]],
    client_id: int,
    loss: float,
    learning_rate: float,
    output_path: str | Path,
) -> Path:
    """Export the sampled net.0.weight[0,0] Adam trace for RISC Zero."""

    if not training_trace:
        raise ValueError("training_trace must not be empty")

    payload = {
        "parameter": "net.0.weight[0,0]",
        "client_id": client_id,
        "initial_weight": training_trace[0]["weight_before"],
        "gradients": [
            row["gradient"]
            for row in training_trace
        ],
        "pytorch_final_weight": training_trace[-1]["weight_after"],
        "learning_rate": learning_rate,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "steps": len(training_trace),
        "loss": float(loss),
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return path


def run_risc0_verification(
    trace_path: str | Path,
    host_path: str | Path = "zkvm/target/release/host",
) -> dict[str, Any]:
    """Run the RISC Zero host and parse its machine-readable result."""

    result = subprocess.run(
        [str(host_path), str(trace_path)],
        capture_output=True,
        text=True,
        check=True,
    )

    result_line = None

    for line in result.stdout.splitlines():
        if line.startswith("ZK_RESULT "):
            result_line = line
            break

    if result_line is None:
        raise RuntimeError(
            "RISC Zero host did not emit ZK_RESULT"
        )

    fields = {}

    for item in result_line.removeprefix("ZK_RESULT ").split():
        key, value = item.split("=", 1)
        fields[key] = value

    return {
        "verified": True,
        "proved_final_weight": float(
            fields["proved_final_weight"]
        ),
        "proving_time_ms": int(
            fields["proving_time_ms"]
        ),
        "verification_time_ms": int(
            fields["verification_time_ms"]
        ),
        "stdout": result.stdout,
    }

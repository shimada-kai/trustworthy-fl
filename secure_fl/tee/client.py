from collections.abc import Mapping, Sequence
from typing import Any

import requests
import torch


def aggregate_via_tee_api(
    client_states: Sequence[Mapping[str, torch.Tensor]],
    api_url: str = "http://127.0.0.1:8000",
) -> dict[str, torch.Tensor]:
    """Send client models to the TEE Aggregator API and receive the median."""

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
        timeout=60,
    )
    response.raise_for_status()

    aggregated_state = response.json()["aggregated_state"]

    return {
        name: torch.tensor(value)
        for name, value in aggregated_state.items()
    }
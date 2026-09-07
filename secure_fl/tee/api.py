from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from secure_fl.tee.aggregator import aggregate_in_tee


app = FastAPI(title="TDX Aggregator API")


class AggregateRequest(BaseModel):
    client_states: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/aggregate")
def aggregate(request: AggregateRequest) -> dict[str, Any]:
    client_states = []

    for state in request.client_states:
        tensor_state = {
            name: torch.tensor(value)
            for name, value in state.items()
        }
        client_states.append(tensor_state)

    aggregated_state = aggregate_in_tee(client_states)

    return {
        "aggregated_state": {
            name: tensor.tolist()
            for name, tensor in aggregated_state.items()
        }
    }
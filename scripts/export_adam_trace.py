import json
from pathlib import Path

import torch

from secure_fl.dataset.adult import get_input_dim, load_client_data
from secure_fl.model.mlp import AdultMLP
from secure_fl.model.train import train_model_with_trace

SEED = 42
NUM_CLIENTS = 7
CLIENT_ID = 0
BATCH_SIZE = 256
LR = 0.001
EPOCHS = 1

torch.manual_seed(SEED)

trainloader, _ = load_client_data(
    partition_id=CLIENT_ID,
    num_partitions=NUM_CLIENTS,
    batch_size=BATCH_SIZE,
    seed=SEED,
)

model = AdultMLP(get_input_dim(SEED))

loss, trace = train_model_with_trace(
    model=model,
    trainloader=trainloader,
    epochs=EPOCHS,
    lr=LR,
    device=torch.device("cpu"),
)

payload = {
    "parameter": "net.0.weight[0,0]",
    "client_id": CLIENT_ID,
    "initial_weight": trace[0]["weight_before"],
    "gradients": [row["gradient"] for row in trace],
    "pytorch_final_weight": trace[-1]["weight_after"],
    "learning_rate": LR,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "steps": len(trace),
    "loss": loss,
}

out = Path("zkvm/adam_trace.json")
out.write_text(json.dumps(payload, indent=2))

print(f"Saved: {out}")
print(f"Parameter: {payload['parameter']}")
print(f"Steps: {payload['steps']}")
print(f"Initial weight: {payload['initial_weight']:.10f}")
print(f"PyTorch final weight: {payload['pytorch_final_weight']:.10f}")

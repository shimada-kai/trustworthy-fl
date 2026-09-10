import torch

from secure_fl.dataset.adult import get_input_dim, load_client_data
from secure_fl.model.mlp import AdultMLP
from secure_fl.model.train import train_model_with_trace

SEED = 42
NUM_CLIENTS = 7
BATCH_SIZE = 256
LR = 0.001
EPOCHS = 1
CLIENT_ID = 0

torch.manual_seed(SEED)

device = torch.device("cpu")

trainloader, _ = load_client_data(
    partition_id=CLIENT_ID,
    num_partitions=NUM_CLIENTS,
    batch_size=BATCH_SIZE,
    seed=SEED,
)

input_dim = get_input_dim(SEED)
model = AdultMLP(input_dim)

loss, trace = train_model_with_trace(
    model=model,
    trainloader=trainloader,
    epochs=EPOCHS,
    lr=LR,
    device=device,
)

print(f"client_id: {CLIENT_ID}")
print(f"input_dim: {input_dim}")
print(f"steps: {len(trace)}")
print(f"loss: {loss:.8f}")
print()

for row in trace:
    print(
        f"step={row['step']:02d} "
        f"w_before={row['weight_before']:+.10f} "
        f"grad={row['gradient']:+.10f} "
        f"m_before={row['exp_avg_before']:+.10f} "
        f"v_before={row['exp_avg_sq_before']:+.10f} "
        f"w_after={row['weight_after']:+.10f} "
        f"m_after={row['exp_avg_after']:+.10f} "
        f"v_after={row['exp_avg_sq_after']:+.10f}"
    )

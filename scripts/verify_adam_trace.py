import math
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

BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8

torch.manual_seed(SEED)

device = torch.device("cpu")

trainloader, _ = load_client_data(
    partition_id=CLIENT_ID,
    num_partitions=NUM_CLIENTS,
    batch_size=BATCH_SIZE,
    seed=SEED,
)

model = AdultMLP(get_input_dim(SEED))

_, trace = train_model_with_trace(
    model=model,
    trainloader=trainloader,
    epochs=EPOCHS,
    lr=LR,
    device=device,
)

all_ok = True

for row in trace:
    t = row["step"] + 1

    w_prev = row["weight_before"]
    g = row["gradient"]
    m_prev = row["exp_avg_before"]
    v_prev = row["exp_avg_sq_before"]

    expected_m = BETA1 * m_prev + (1.0 - BETA1) * g
    expected_v = BETA2 * v_prev + (1.0 - BETA2) * (g * g)

    m_hat = expected_m / (1.0 - BETA1 ** t)
    v_hat = expected_v / (1.0 - BETA2 ** t)

    expected_w = w_prev - LR * m_hat / (math.sqrt(v_hat) + EPS)

    m_error = abs(expected_m - row["exp_avg_after"])
    v_error = abs(expected_v - row["exp_avg_sq_after"])
    w_error = abs(expected_w - row["weight_after"])

    ok = (
        m_error < 1e-8
        and v_error < 1e-10
        and w_error < 1e-6
    )

    all_ok &= ok

    print(
        f"step={row['step']:02d} "
        f"m_err={m_error:.3e} "
        f"v_err={v_error:.3e} "
        f"w_err={w_error:.3e} "
        f"{'OK' if ok else 'FAIL'}"
    )

print()
print("Adam trace verification:", "SUCCESS" if all_ok else "FAILED")

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def train_model(
    model,
    trainloader,
    epochs: int,
    lr: float,
    device: torch.device,
):
    model.to(device)
    model.train()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    total_examples = 0

    for _ in range(epochs):
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = y.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate_model(model, dataloader, device: torch.device):
    model.to(device)
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_examples = 0
    probs_all = []
    targets_all = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        probs_all.append(probs.cpu().numpy())
        targets_all.append(y.cpu().numpy())

    probs = np.concatenate(probs_all)
    targets = np.concatenate(targets_all).astype(int)
    preds = (probs >= 0.5).astype(int)

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": float(accuracy_score(targets, preds)),
        "roc_auc": float(roc_auc_score(targets, probs)),
        "pr_auc": float(average_precision_score(targets, probs)),
    }


def train_model_with_trace(
    model,
    trainloader,
    epochs: int,
    lr: float,
    device: torch.device,
):
    """
    Sampled ZK Verification PoC用のtraining traceを記録する。

    既存のtrain_model()は変更せず、
    net.0.weight[0, 0]のAdam更新軌跡を各optimizer stepで記録する。

    現在のPoCではgradientをRISC Zeroへの入力として使用するため、
    forward/backwardからgradientが生成された過程そのものは
    ZK proofの対象外である。
    """
    model.to(device)
    model.train()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    total_examples = 0

    trace = []
    step = 0

    target_parameter = dict(model.named_parameters())["net.0.weight"]

    for _ in range(epochs):
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            weight_before = float(
                target_parameter[0, 0].detach().cpu().item()
            )

            state_before = optimizer.state.get(target_parameter, {})

            exp_avg_before = float(
                state_before["exp_avg"][0, 0].detach().cpu().item()
            ) if "exp_avg" in state_before else 0.0

            exp_avg_sq_before = float(
                state_before["exp_avg_sq"][0, 0].detach().cpu().item()
            ) if "exp_avg_sq" in state_before else 0.0

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            gradient = float(
                target_parameter.grad[0, 0].detach().cpu().item()
            )

            optimizer.step()

            weight_after = float(
                target_parameter[0, 0].detach().cpu().item()
            )

            state_after = optimizer.state[target_parameter]

            exp_avg_after = float(
                state_after["exp_avg"][0, 0].detach().cpu().item()
            )

            exp_avg_sq_after = float(
                state_after["exp_avg_sq"][0, 0].detach().cpu().item()
            )

            trace.append(
                {
                    "step": step,
                    "weight_before": weight_before,
                    "gradient": gradient,
                    "exp_avg_before": exp_avg_before,
                    "exp_avg_sq_before": exp_avg_sq_before,
                    "weight_after": weight_after,
                    "exp_avg_after": exp_avg_after,
                    "exp_avg_sq_after": exp_avg_sq_after,
                }
            )

            step += 1

            batch_size = y.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    avg_loss = total_loss / max(total_examples, 1)

    return avg_loss, trace

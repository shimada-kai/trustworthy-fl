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

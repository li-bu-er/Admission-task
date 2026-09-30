import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models import build_graph_model
from utils import accuracy, get_device, mae, timer


def evaluate(model, loader, device, task):
    model.eval()
    total_score = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logtis = model(batch.x, batch.edge_index, batch.batch)
            labels = batch.y.view(-1)
            n = labels.numel()
            if task == "classification":
                total_score += accuracy(logtis, labels) * n
            else:
                total_score += mae(logtis, labels) * n
            total_examples += n
    return total_score / max(total_examples, 1)


def train_one_epoch(model, loader, optimizer, device, task):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)

        if task == "classification":
            loss = F.cross_entropy(out, batch.y.view(-1))
            n = batch.num_graphs
        else:
            loss = F.l1_loss(out.view(-1), batch.y.view(-1))
            n = batch.num_graphs

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * n
        total_examples += n

    return total_loss / max(total_examples, 1)


def run_experiment(
    dataset_name,
    train_set,
    val_set,
    test_set,
    task,
    model_name,
    pool,
    in_dim,
    hidden_dim,
    embed_dim,
    num_classes,
    num_layers,
    heads,
    dropout,
    lr,
    weight_decay,
    epochs,
    batch_size,
    device,
    verbose,
):
    device = device or get_device()

    model = build_graph_model(
        name=model_name,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_classes=num_classes,
        dropout=dropout,
        num_layers=num_layers,
        heads=heads,
        pool=pool,
        task=task,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    best_val = float("inf") if task == "regression" else 0.0
    best_test = best_val
    best_epoch = 0
    history = []
    metric_name = "val_mae" if task == "regression" else "val_acc"

    with timer() as t:
        epoch_iter = range(1, epochs + 1)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc=f"{model_name}-{pool}")
        for epoch in epoch_iter:
            loss = train_one_epoch(model, train_loader, optimizer, device, task)
            val_score = evaluate(model, val_loader, device, task)
            test_score = evaluate(model, test_loader, device, task)

            if task == "classification":
                train_score = evaluate(model, train_loader, device, task)
                record = {
                    "epoch": epoch,
                    "loss": loss,
                    "train_acc": train_score,
                    "val_acc": val_score,
                    "test_acc": test_score,
                }
                improved = val_score >= best_val
            else:
                record = {
                    "epoch": epoch,
                    "loss": loss,
                    "val_mae": val_score,
                    "test_mae": test_score,
                }
                improved = val_score <= best_val

            history.append(record)

            if improved:
                best_val = val_score
                best_test = test_score
                best_epoch = epoch

    result = {
        "dataset": dataset_name,
        "task": task,
        "model": model_name,
        "pool": pool,
        "hidden_dim": hidden_dim,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "lr": lr,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        metric_name: best_val,
        "best_test_acc" if task == "classification" else "best_test_mae": best_test,
        "train_time_sec": t["elapsed"],
        "device": str(device),
    }
    if verbose:
        if task == "classification":
            print(
                f"[{dataset_name}|{model_name}|{pool}] "
                f"best_val={best_val:.4f} test@{best_epoch}={best_test:.4f} "
                f"time={t['elapsed']:.2f}s"
            )
        else:
            print(
                f"[{dataset_name}|{model_name}|{pool}] "
                f"best_val_mae={best_val:.4f} test_mae@{best_epoch}={best_test:.4f} "
                f"time={t['elapsed']:.2f}s"
            )

    result["history"] = history
    return result

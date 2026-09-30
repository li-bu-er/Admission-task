
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from models import build_model
from utils import accuracy, timer, get_device


def evaluate_full(model,data,mask,device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        acc = accuracy(logits[mask], data.y[mask])
    return acc


def train_full_graph(model, data, optimizer, device):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


def make_neighbor_loader(data, input_mask, num_layers, batch_size, num_neighbors, shuffle=True):
    # NeighborLoader 需要每层采样数；None 时默认每层采 15 个邻居
    if num_neighbors is None:
        num_neighbors = [15] * num_layers
    return NeighborLoader(
        data,
        input_nodes=input_mask,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        shuffle=shuffle,
        num_workers=0,
    )


def train_one_epoch_sampled(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(
            logits[: batch.batch_size], batch.y[: batch.batch_size]
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
        total_examples += batch.batch_size
    return total_loss / max(total_examples, 1)


def run_experiment(
    dataset_name,
    data,
    num_features,
    num_classes,
    model_name,
    mode,
    hidden_dim,
    num_layers,
    heads,
    dropout,
    lr,
    weight_decay,
    epochs,
    batch_size,
    num_neighbors,
    device,
    verbose
):
    assert mode in ("full", "sample")
    device = device or get_device()

    model = build_model(name=model_name,in_dim=num_features,hidden_dim=hidden_dim,out_dim=num_classes,dropout=dropout,num_layers=num_layers,heads=heads).to(device)  

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = None
    if mode == "sample":
        loader = make_neighbor_loader(data=data,input_mask=data.train_mask,num_layers=num_layers,batch_size=batch_size,num_neighbors=num_neighbors)

    best_val = 0.0
    best_test = 0.0
    best_epoch = 0
    history = []

    with timer() as t:
        epoch_iter = range(1, epochs + 1)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc=f"{model_name}-{mode}")
        for epoch in epoch_iter:
            if mode == "full":
                loss = train_full_graph(model, data, optimizer, device)
            else:
                loss = train_one_epoch_sampled(model, loader, optimizer, device)

            val_acc = evaluate_full(model, data, data.val_mask, device)
            test_acc = evaluate_full(model, data, data.test_mask, device)
            train_acc = evaluate_full(model, data, data.train_mask, device)

            history.append({"epoch": epoch,"loss": loss,"train_acc": train_acc,"val_acc": val_acc,"test_acc": test_acc,})

            if val_acc >= best_val:
                best_val = val_acc
                best_test = test_acc
                best_epoch = epoch

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "mode": mode,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "lr": lr,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size if mode == "sample" else None,
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
        "best_test_acc": best_test,
        "train_time_sec": t["elapsed"],
        "device": str(device)
    }
    if verbose:
        print(
            f"[{dataset_name}|{model_name}|{mode}] "
            f"best_val={best_val:.4f} test@{best_epoch}={best_test:.4f} "
            f"time={t['elapsed']:.2f}s"
        )

    result["history"] = history
    return result

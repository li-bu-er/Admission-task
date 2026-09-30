import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm

from models import build_link_model
from utils import get_device, link_scores_to_metrics, timer


def evaluate(model, data, device):
    """在 val/test 划分上评估；消息传递使用 data.edge_index（训练图）。"""
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index) #[N, D]
        logits = model.decode(z, data.edge_label_index) #logits 形状是 [E]，E = edge_label_index.size(1)
        scores = torch.sigmoid(logits)
        auc, ap = link_scores_to_metrics(data.edge_label, scores)
    return auc, ap


def train_full_graph(model, train_data, optimizer, device):
    model.train()
    train_data = train_data.to(device)
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    logits = model.decode(z, train_data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(
        logits, train_data.edge_label.float()
    )
    loss.backward()
    optimizer.step()
    return float(loss.item())


def make_link_loader(train_data, num_layers, batch_size, num_neighbors, shuffle=True):
    if num_neighbors is None:
        num_neighbors = [15] * num_layers
    return LinkNeighborLoader(
        train_data,
        edge_label_index=train_data.edge_label_index,
        edge_label=train_data.edge_label,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        neg_sampling_ratio=1.0,
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
        z = model.encode(batch.x, batch.edge_index)
        logits = model.decode(z, batch.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(
            logits, batch.edge_label.float()
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.edge_label.numel()
        total_examples += batch.edge_label.numel()

    return total_loss / max(total_examples, 1)


def run_experiment(
    dataset_name,
    train_data,
    val_data,
    test_data,
    model_name,
    mode,
    in_dim,
    hidden_dim,
    out_dim,
    num_layers,
    heads,
    dropout,
    lr,
    weight_decay,
    epochs,
    batch_size,
    num_neighbors,
    device,
    verbose,
):
    assert mode in ("full", "sample")
    device = device or get_device()

    model = build_link_model(
        name=model_name,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=dropout,
        num_layers=num_layers,
        heads=heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = None
    if mode == "sample":
        loader = make_link_loader(train_data=train_data,num_layers=num_layers,batch_size=batch_size,num_neighbors=num_neighbors,)

    best_val_auc = 0.0
    best_test_auc = 0.0
    best_test_ap = 0.0
    best_val_ap = 0.0
    best_epoch = 0
    history = []

    with timer() as t:
        epoch_iter = range(1, epochs + 1)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc=f"{model_name}-{mode}")
        for epoch in epoch_iter:
            if mode == "full":
                loss = train_full_graph(model, train_data, optimizer, device)
            else:
                loss = train_one_epoch_sampled(model, loader, optimizer, device)

            val_auc, val_ap = evaluate(model, val_data, device)
            test_auc, test_ap = evaluate(model, test_data, device)

            history.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                    "test_auc": test_auc,
                    "test_ap": test_ap,
                }
            )

            if val_auc >= best_val_auc:
                best_val_auc = val_auc
                best_val_ap = val_ap
                best_test_auc = test_auc
                best_test_ap = test_ap
                best_epoch = epoch

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "mode": mode,
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "num_layers": num_layers,
        "lr": lr,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size if mode == "sample" else None,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_val_ap": best_val_ap,
        "best_test_auc": best_test_auc,
        "best_test_ap": best_test_ap,
        "train_time_sec": t["elapsed"],
        "device": str(device),
    }
    if verbose:
        print(
            f"[{dataset_name}|{model_name}|{mode}] "
            f"best_val_auc={best_val_auc:.4f} test_auc@{best_epoch}={best_test_auc:.4f} "
            f"test_ap={best_test_ap:.4f} time={t['elapsed']:.2f}s"
        )

    result["history"] = history
    return result

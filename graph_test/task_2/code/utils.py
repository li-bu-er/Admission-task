from contextlib import contextmanager
import csv
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.datasets import Flickr, Planetoid
from torch_geometric.transforms import NormalizeFeatures, RandomLinkSplit


from hyperparams import DEFAULTS

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def set_seed(seed=None):
    import random
    import numpy as np

    seed = DEFAULTS["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_link_dataset(name, root=None, val_ratio=0.05, test_ratio=0.10, seed=None):
    """加载数据集并划分链路，返回 (train_data, val_data, test_data, num_features)。"""
    seed = DEFAULTS["seed"] if seed is None else seed
    root = Path(root) if root is not None else DEFAULT_DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)

    key = name.lower()
    if key in ("cora", "citeseer"):
        name_map = {"cora": "Cora", "citeseer": "CiteSeer"}
        dataset = Planetoid(
            root=str(root),
            name=name_map[key],
            transform=NormalizeFeatures(),
        )
    elif key == "flickr":
        dataset = Flickr(
            root=str(root / "Flickr"),
            transform=NormalizeFeatures(),
        )
    else:
        raise ValueError(f"未知数据集: {name}. 可选: cora, citeseer, flickr")

    data = dataset[0]
    num_features = dataset.num_features

    split = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = split(data)

    print("=" * 50)
    print(f"数据集: {name} (链路预测)")
    print(f"  节点数           : {data.num_nodes}")
    print(f"  原始边数         : {data.num_edges}")
    print(f"  特征维度         : {num_features}")
    print(f"  消息传递边数     : {train_data.num_edges}")
    print("=" * 50)

    return train_data, val_data, test_data, num_features


def link_scores_to_metrics(labels, scores):
    labels = labels.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    if len(set(labels.tolist())) < 2:
        return 0.0, 0.0
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return float(auc), float(ap)


@contextmanager
def timer():
    info = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield info
    finally:
        info["elapsed"] = time.perf_counter() - start


def get_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_csv(row, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

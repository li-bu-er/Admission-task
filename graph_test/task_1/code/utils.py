from contextlib import contextmanager
import time
import csv
import json
from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.transforms import NormalizeFeatures


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


def load_dataset(name, root=None):
    """加载 Cora / CiteSeer / Flickr，返回 (data, num_features, num_classes)。"""
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
    num_classes = dataset.num_classes

    print("=" * 50)
    print(f"数据集: {name}")
    print(f"  节点数       : {data.num_nodes}")
    print(f"  边数         : {data.num_edges}")
    print(f"  特征维度     : {num_features}")
    print(f"  类别数       : {num_classes}")
    print(f"  训练节点数   : {int(data.train_mask.sum())}")
    print(f"  验证节点数   : {int(data.val_mask.sum())}")
    print(f"  测试节点数   : {int(data.test_mask.sum())}")
    print("=" * 50)

    return data, num_features, num_classes


def accuracy(logits, labels):
    if logits.dim() > 1:
        logits = logits.argmax(dim=-1)
    correct = (logits == labels).sum().item()
    acc = correct / max(labels.numel(), 1)
    return acc


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
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


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

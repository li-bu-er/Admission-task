from contextlib import contextmanager
import csv
import json
import os
import shutil
import ssl
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, ZINC
from torch_geometric.transforms import Constant


from hyperparams import DEFAULTS

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DEFAULT_TU_DATASETS = ["MUTAG", "PROTEINS", "IMDB-BINARY"]


def set_seed(seed=None):
    import random

    seed = DEFAULTS["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

ZINC_MOLECULE_URLS = [
    "https://dl.dropboxusercontent.com/s/feo9qle74kg48gy/molecules.zip",
    "https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1",
]
ZINC_INDEX_URL = (
    "https://raw.githubusercontent.com/graphdeeplearning/"
    "benchmarking-gnns/master/data/molecules/{}.index"
)
ZINC_RAW_FILES = [
    "train.pickle",
    "val.pickle",
    "test.pickle",
    "train.index",
    "val.index",
    "test.index",
]


def download_file(url, dest, timeout=300):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)


def zinc_raw_ready(raw_dir):
    raw_dir = Path(raw_dir)
    return all((raw_dir / name).exists() for name in ZINC_RAW_FILES)


def prepare_zinc_dataset(root):
    """
    确保 ZINC 原始文件就绪。若 Dropbox SSL 失败，可手动解压到 root/raw/ 后重试。
    需要文件：train/val/test.pickle 与 train/val/test.index
    """
    root = Path(root)
    raw_dir = root / "raw"
    processed_train = root / "subset" / "processed" / "train.pt"

    if processed_train.exists() or zinc_raw_ready(raw_dir):
        return

    print("ZINC 本地数据未找到，尝试下载 molecules.zip ...")
    zip_path = root / "molecules.zip"
    last_err = None
    for url in ZINC_MOLECULE_URLS:
        try:
            print(f"  下载: {url}")
            download_file(url, zip_path)
            last_err = None
            break
        except Exception as e:
            last_err = e
            print(f"  失败: {e}")

    if last_err is not None:
        raise RuntimeError(
            "ZINC 自动下载失败（多为网络/SSL）。请手动操作：\n"
            "1. 浏览器下载 molecules.zip：\n"
            "   https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1\n"
            f"2. 解压后将 train/val/test.pickle 放到：{raw_dir}\n"
            "3. 从 GitHub 下载 train.index / val.index / test.index 到同一目录：\n"
            "   https://github.com/graphdeeplearning/benchmarking-gnns/tree/master/data/molecules\n"
            "4. 再运行 python run_experiments.py"
        ) from last_err

    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    molecules_dir = root / "molecules"
    if molecules_dir.exists():
        for item in molecules_dir.iterdir():
            shutil.move(str(item), str(raw_dir / item.name))
        molecules_dir.rmdir()
    os.remove(zip_path)

    for split in ("train", "val", "test"):
        index_path = raw_dir / f"{split}.index"
        if not index_path.exists():
            print(f"  下载 index: {split}")
            download_file(ZINC_INDEX_URL.format(split), index_path)

    if not zinc_raw_ready(raw_dir):
        raise RuntimeError(f"ZINC 数据仍不完整，请检查目录: {raw_dir}")


def atom_type_one_hot(data, num_types=28):
    """ZINC 节点特征是原子类型 Long，转为 one-hot float 供 GNN 使用。"""
    x = data.x.view(-1).long()
    data.x = F.one_hot(x, num_classes=num_types).to(torch.float)
    return data


def ensure_node_features(dataset):
    """TU 数据集若无节点特征，补常数特征。"""
    if dataset.num_features == 0:
        return TUDataset(dataset.root, dataset.name, transform=Constant(value=1))
    return dataset


def split_tu_dataset(dataset, val_ratio=0.1, test_ratio=0.1, seed=None):
    seed = DEFAULTS["seed"] if seed is None else seed
    labels = dataset.y.view(-1).cpu().numpy()
    indices = np.arange(len(dataset))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(indices, labels))

    train_val_labels = labels[train_val_idx]
    relative_test = test_ratio
    val_size = val_ratio / (1.0 - relative_test)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss2.split(train_val_idx, train_val_labels))

    train_set = dataset[train_val_idx[train_idx].tolist()]
    val_set = dataset[train_val_idx[val_idx].tolist()]
    test_set = dataset[test_idx.tolist()]
    return train_set, val_set, test_set


def load_graph_dataset(name, root=None, seed=None):
    """
    加载图级数据集。
    返回 (train_set, val_set, test_set, num_features, num_classes, task, display_name)
    """
    seed = DEFAULTS["seed"] if seed is None else seed
    root = Path(root) if root is not None else DEFAULT_DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    key = name.lower()

    if key == "zinc":
        zinc_root = root / "ZINC"
        prepare_zinc_dataset(zinc_root)
        train_set = ZINC(
            root=str(zinc_root), subset=True, split="train", transform=atom_type_one_hot
        )
        val_set = ZINC(
            root=str(zinc_root), subset=True, split="val", transform=atom_type_one_hot
        )
        test_set = ZINC(
            root=str(zinc_root), subset=True, split="test", transform=atom_type_one_hot
        )
        num_features = 28 #ZINC里的原子类型种类数
        num_classes = 1
        task = "regression"
        display_name = "ZINC"

        print("=" * 50)
        print("数据集: ZINC (图级回归)")
        print(f"  训练图数       : {len(train_set)}")
        print(f"  验证图数       : {len(val_set)}")
        print(f"  测试图数       : {len(test_set)}")
        print(f"  节点特征维度   : {num_features} (原子类型 one-hot)")
        print(f"  任务类型       : regression (MAE)")
        print("=" * 50)
        return train_set, val_set, test_set, num_features, num_classes, task, display_name

    tu_name = name.upper() if key != "tu" else "MUTAG"
    if key == "imdb-binary":
        tu_name = "IMDB-BINARY"
    dataset = TUDataset(root=str(root / "TUDataset"), name=tu_name)
    dataset = ensure_node_features(dataset)
    train_set, val_set, test_set = split_tu_dataset(dataset, seed=seed)
    num_features = dataset.num_features
    num_classes = int(dataset.num_classes)
    task = "classification"
    display_name = tu_name

    print("=" * 50)
    print(f"数据集: TUDataset / {tu_name} (图分类)")
    print(f"  总图数         : {len(dataset)}")
    print(f"  训练/验证/测试 : {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print(f"  节点特征维度   : {num_features}")
    print(f"  类别数         : {num_classes}")
    print("=" * 50)
    return train_set, val_set, test_set, num_features, num_classes, task, display_name


def accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return correct / max(labels.numel(), 1)


def mae(pred, target):
    return float(torch.mean(torch.abs(pred.view(-1) - target.view(-1))).item())


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

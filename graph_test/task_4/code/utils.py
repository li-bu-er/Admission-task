from contextlib import contextmanager
import csv
import json
import shutil
import ssl
import time
import urllib.request
from pathlib import Path

import torch


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


KG_DATASETS = {
    "fb15k-237": {
        "display": "FB15k-237",
        "urls": [
            (
                "https://raw.githubusercontent.com/DeepGraphLearning/"
                "KnowledgeGraphEmbedding/master/data/FB15k-237/{}.txt"
            ),
            (
                "https://cdn.jsdelivr.net/gh/DeepGraphLearning/"
                "KnowledgeGraphEmbedding@master/data/FB15k-237/{}.txt"
            ),
        ],
    },
    "wn18rr": {
        "display": "WN18RR",
        "urls": [
            (
                "https://raw.githubusercontent.com/DeepGraphLearning/"
                "KnowledgeGraphEmbedding/master/data/wn18rr/{}.txt"
            ),
            (
                "https://cdn.jsdelivr.net/gh/DeepGraphLearning/"
                "KnowledgeGraphEmbedding@master/data/wn18rr/{}.txt"
            ),
        ],
    },
}


def download_file(url, dest, timeout=120):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)


def read_triples(path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def prepare_kg_files(name, root=None):
    root = Path(root) if root is not None else DEFAULT_DATA_ROOT
    key = name.lower()
    if key not in KG_DATASETS:
        raise ValueError(f"未知数据集: {name}. 可选: {list(KG_DATASETS.keys())}")

    info = KG_DATASETS[key]
    data_dir = root / key
    data_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        path = data_dir / f"{split}.txt"
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"  下载 {split}.txt ...")
        last_err = None
        for tmpl in info["urls"]:
            url = tmpl.format(split)
            try:
                download_file(url, path)
                if path.stat().st_size > 0:
                    last_err = None
                    break
            except Exception as e:
                last_err = e
                print(f"    失败: {url}\n      ({e})")
        if last_err is not None:
            raise RuntimeError(
                f"下载 {name}/{split}.txt 失败（网络/DNS 不可用）。\n"
                f"请用浏览器打开下面链接，另存为到：\n  {path}\n"
                f"链接（任选一个能打开的）：\n"
                + "\n".join(f"  - {tmpl.format(split)}" for tmpl in info["urls"])
                + "\n格式: head\\trelation\\ttail（每行一个三元组）"
            ) from last_err
    return data_dir


def load_kg_dataset(name, root=None):
    """
    加载知识图谱，返回 KGData 对象。
    划分：train / valid / test 三个 txt 文件。
    """
    data_dir = prepare_kg_files(name, root=root)
    key = name.lower()
    display = KG_DATASETS[key]["display"]

    train_triples = read_triples(data_dir / "train.txt")
    valid_triples = read_triples(data_dir / "valid.txt")
    test_triples = read_triples(data_dir / "test.txt")

    entities = set()
    relations = set()
    for triples in (train_triples, valid_triples, test_triples):
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)

    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}

    def to_ids(triples):
        return torch.tensor(
            [[entity2id[h], relation2id[r], entity2id[t]] for h, r, t in triples],
            dtype=torch.long,
        )

    train = to_ids(train_triples)
    valid = to_ids(valid_triples)
    test = to_ids(test_triples)
    all_triples = torch.cat([train, valid, test], dim=0)

    hr_to_tails = {}
    rt_to_heads = {}
    for h, r, t in all_triples.tolist():
        hr_to_tails.setdefault((h, r), set()).add(t)
        rt_to_heads.setdefault((r, t), set()).add(h)

    print("=" * 50)
    print(f"数据集: {display} (知识图谱补全)")
    print(f"  实体数         : {len(entity2id)}")
    print(f"  关系数         : {len(relation2id)}")
    print(f"  训练三元组     : {len(train)}")
    print(f"  验证三元组     : {len(valid)}")
    print(f"  测试三元组     : {len(test)}")
    print("=" * 50)

    return KGData(
        name=display,
        train=train,
        valid=valid,
        test=test,
        num_entities=len(entity2id),
        num_relations=len(relation2id),
        hr_to_tails=hr_to_tails,
        rt_to_heads=rt_to_heads,
    )


class KGData:
    def __init__(
        self,
        name,
        train,
        valid,
        test,
        num_entities,
        num_relations,
        hr_to_tails,
        rt_to_heads,
    ):
        self.name = name
        self.train = train
        self.valid = valid
        self.test = test
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hr_to_tails = hr_to_tails
        self.rt_to_heads = rt_to_heads


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

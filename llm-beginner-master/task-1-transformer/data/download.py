"""下载 ChnSentiCorp 二分类到当前目录。

用 lansinuote/ChnSentiCorp（label 仅 0/1）。
别用 Kerwin11（三分类）或 seamew（依赖 Google Drive）。
国内镜像：HF_ENDPOINT=https://hf-mirror.com
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
CACHE_DIR = Path.home() / ".cache" / "llm-beginner" / "chnsenticorp_bin"


def _normalize_dataset(ds):
    from datasets import DatasetDict

    def _one(split_ds):
        cols = set(split_ds.column_names)
        rename = {}
        if "text" not in cols:
            for c in ("review", "sentence", "content", "raw_text"):
                if c in cols:
                    rename[c] = "text"
                    break
        if "label" not in cols:
            for c in ("labels", "sentiment", "polarity"):
                if c in cols:
                    rename[c] = "label"
                    break
        if rename:
            split_ds = split_ds.rename_columns(rename)
        if "text" not in split_ds.column_names or "label" not in split_ds.column_names:
            raise ValueError(f"缺少 text/label 列，实际: {split_ds.column_names}")
        split_ds = split_ds.select_columns(["text", "label"])
        labels = set(int(x) for x in split_ds["label"])
        if labels - {0, 1}:
            raise ValueError(f"期望二分类标签 {{0,1}}，实际出现 {sorted(labels)}")
        return split_ds

    if hasattr(ds, "keys"):
        return DatasetDict({k: _one(ds[k]) for k in ds.keys()})
    return _one(ds)


def _load_from_hf():
    from datasets import load_dataset

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("下载 lansinuote/ChnSentiCorp（二分类 parquet）...")
    return load_dataset("lansinuote/ChnSentiCorp", cache_dir=str(CACHE_DIR))


def _export_parquet(ds) -> None:
    for split in ds.keys():
        out = DATA_DIR / f"{split}.parquet"
        split_ds = ds[split]
        if not hasattr(split_ds, "to_parquet"):
            from datasets import Dataset

            split_ds = Dataset.from_list(list(split_ds))
        split_ds.to_parquet(str(out))
        labels = sorted(set(int(x) for x in split_ds["label"]))
        print(f"  {split}: {len(split_ds)} 条, labels={labels} -> {out.name}")


def main():
    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 国内建议先设置镜像：")
        print('  PowerShell: $env:HF_ENDPOINT = "https://hf-mirror.com"')
        print("  CMD:        set HF_ENDPOINT=https://hf-mirror.com\n")

    try:
        import datasets  # noqa: F401
    except ImportError:
        sys.exit("[错误] 缺少依赖：pip install datasets pyarrow")

    try:
        ds = _normalize_dataset(_load_from_hf())
    except Exception as e:
        sys.exit(
            f"[错误] 下载失败：{e}\n"
            "请确认已设置 HF_ENDPOINT=https://hf-mirror.com 后重试。"
        )

    print(f"\n导出到 {DATA_DIR} ...")
    _export_parquet(ds)
    print("\n完成。")


if __name__ == "__main__":
    main()

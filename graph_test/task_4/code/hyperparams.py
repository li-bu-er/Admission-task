"""从 hyperparams.yaml 加载任务四超参数。"""

from pathlib import Path

import yaml

with open(Path(__file__).with_name("hyperparams.yaml"), "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

DEFAULTS = _cfg["defaults"]
DATASETS = _cfg["datasets"]
MODELS = _cfg["models"]
PARAM_LRS = _cfg["param_lrs"]
PARAM_EMBED_DIMS = _cfg["param_embed_dims"]

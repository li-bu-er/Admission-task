"""从 hyperparams.yaml 加载任务三超参数。"""

from pathlib import Path

import yaml

with open(Path(__file__).with_name("hyperparams.yaml"), "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

DEFAULTS = _cfg["defaults"]
DATASETS = _cfg["datasets"]
MODELS = _cfg["models"]
POOLS = _cfg["pools"]
PARAM_POOLS = _cfg["param_pools"]
PARAM_LRS = _cfg["param_lrs"]

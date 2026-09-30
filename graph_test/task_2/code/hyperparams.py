"""从 hyperparams.yaml 加载任务二超参数。"""

from pathlib import Path

import yaml

with open(Path(__file__).with_name("hyperparams.yaml"), "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

DEFAULTS = _cfg["defaults"]
DATASETS = _cfg["datasets"]
MODELS = _cfg["models"]
MODES = _cfg["modes"]
PARAM_LRS = _cfg["param_lrs"]
PARAM_NUM_LAYERS = _cfg["param_num_layers"]

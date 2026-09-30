# 任务四：知识图谱补全（KGE）

task_4/
├── code/
│   ├── hyperparams.yaml    # 超参数配置
│   ├── utils.py            # 数据下载/加载、KGData、存结果
│   ├── models.py           # TransE / RotatE / ConvE
│   ├── train.py            # 训练、过滤评测、run_experiment
│   ├── train_kg.py         # 单次实验
│   └── run_experiments.py  # 批量实验
├── data/                   # 自动下载缓存
└── results/

# 完整实验
python run_experiments.py

超参数见 `code/hyperparams.yaml`。结果写入 `task_4/results/summary.csv`。

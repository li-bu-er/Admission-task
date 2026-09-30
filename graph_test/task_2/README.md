# 链路预测

task_2/
├── code/
│   ├── hyperparams.yaml    # 超参数配置
│   ├── utils.py            # 读数据、划分边、指标、存结果
│   ├── models.py           # GNN 编码器 + LinkPredictorModel
│   ├── train.py            # full / sample 训练与 run_experiment
│   ├── train_link.py       # 单次实验
│   └── run_experiments.py  # 批量实验
├── data/                   # Cora / CiteSeer / Flickr
└── results/                # summary.csv + json


# 单次实验
python train_link.py --dataset cora --model gcn --mode full --epochs 20
python train_link.py --dataset cora --model gcn --mode sample --epochs 20
....

# 完整实验
python run_experiments.py


Part A：3 数据集 × 4 模型 × 2 模式（全图 / 子图采样）

Part B：Cora + GCN + full，扫描学习率 `{0.001, 0.01, 0.05}` 与层数 `{1,2,3,4}`

超参数见 `code/hyperparams.yaml`。结果写入 `task_2/results/summary.csv`。

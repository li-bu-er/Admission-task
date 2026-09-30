# 任务三：图分类

task_3/
├── code/
│   ├── hyperparams.yaml    # 超参数配置
│   ├── utils.py            # 读数据、划分、指标、存结果
│   ├── models.py           # GNN + 图级池化 + 分类/回归头
│   ├── train.py            # 训练与 run_experiment
│   ├── train_graph.py      # 单次实验
│   └── run_experiments.py  # 批量实验
├── data/                   # 自动下载缓存
└── results/



# 单次实验
python train_graph.py --dataset mutag --model gcn --pool mean --epochs 50
python train_graph.py --dataset zinc --model gcn --pool max --epochs 20
...

# 完整实验
python run_experiments.py

超参数见 `code/hyperparams.yaml`。

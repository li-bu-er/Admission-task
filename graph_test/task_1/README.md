#单个数据集，模型训练
python train_node.py --dataset cora --model gcn --mode full --epochs 20
python train_node.py --dataset citeseer --model gcn --mode full --epochs 20
python train_node.py --dataset flickr --model gcn --mode full --epochs 20
python train_node.py --dataset cora --model gat --mode sample

#全部实验
python run_experiments.py 

超参数见 `code/hyperparams.yaml`。结果查看：`task_1/results/summary.csv`。

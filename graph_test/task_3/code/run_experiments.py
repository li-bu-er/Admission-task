import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hyperparams import DATASETS, DEFAULTS, MODELS, PARAM_LRS, PARAM_POOLS, POOLS
from train import run_experiment
from utils import append_csv, get_device, load_graph_dataset, save_json, set_seed


def parse_args():
    d = DEFAULTS
    p = argparse.ArgumentParser(description="任务三图分类批量实验")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="TUDataset 名称或 zinc",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=["gcn", "gat", "graphsage", "gin"],
    )
    p.add_argument(
        "--pools",
        nargs="+",
        default=POOLS,
        choices=["mean", "avg", "max", "min"],
    )
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--zinc-epochs", type=int, default=d["zinc_epochs"])
    p.add_argument("--hidden", type=int, default=d["hidden"])
    p.add_argument("--embed-dim", type=int, default=d["embed_dim"])
    p.add_argument("--num-layers", type=int, default=d["num_layers"])
    p.add_argument("--heads", type=int, default=d["heads"])
    p.add_argument("--dropout", type=float, default=d["dropout"])
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--skip-param", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=d["seed"])
    return p.parse_args()


def dataset_tag(name):
    return name.lower().replace("-", "")


def pool_tag(pool):
    return "mean" if pool == "avg" else pool


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = results_dir / "summary.csv"

    print("\n" + "#" * 60)
    print("# Part A: 数据集 / 模型 / 池化方法")
    print("#" * 60)

    for ds_name in args.datasets:
        train_set, val_set, test_set, in_dim, num_classes, task, display_name = (
            load_graph_dataset(ds_name, seed=args.seed)
        )
        epochs = args.zinc_epochs if ds_name.lower() == "zinc" else args.epochs

        for model_name in args.models:
            for pool in args.pools:
                tag = f"{dataset_tag(ds_name)}_{model_name}_{pool_tag(pool)}_main"

                set_seed(args.seed)
                result = run_experiment(
                    dataset_name=display_name,
                    train_set=train_set,
                    val_set=val_set,
                    test_set=test_set,
                    task=task,
                    model_name=model_name,
                    pool=pool,
                    in_dim=in_dim,
                    hidden_dim=args.hidden,
                    embed_dim=args.embed_dim,
                    num_classes=num_classes,
                    num_layers=args.num_layers,
                    heads=args.heads,
                    dropout=args.dropout,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    device=device,
                    verbose=True,
                )
                save_json(result, results_dir / f"{tag}.json")
                append_csv(
                    {k: v for k, v in result.items() if k != "history"},
                    summary_csv,
                )

    if not args.skip_param:
        print("\n" + "#" * 60)
        print("# Part B: 池化 / 学习率 敏感性（MUTAG + GCN）")
        print("#" * 60)

        train_set, val_set, test_set, in_dim, num_classes, task, display_name = (
            load_graph_dataset("mutag", seed=args.seed)
        )

        for pool in PARAM_POOLS:
            for lr in PARAM_LRS:
                tag = f"mutag_gcn_{pool_tag(pool)}_param_lr{lr}"

                set_seed(args.seed)
                result = run_experiment(
                    dataset_name=display_name,
                    train_set=train_set,
                    val_set=val_set,
                    test_set=test_set,
                    task=task,
                    model_name="gcn",
                    pool=pool,
                    in_dim=in_dim,
                    hidden_dim=args.hidden,
                    embed_dim=args.embed_dim,
                    num_classes=num_classes,
                    num_layers=args.num_layers,
                    heads=args.heads,
                    dropout=args.dropout,
                    lr=lr,
                    weight_decay=args.weight_decay,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=device,
                    verbose=True,
                )
                save_json(result, results_dir / f"{tag}.json")
                append_csv(
                    {k: v for k, v in result.items() if k != "history"},
                    summary_csv,
                )

    print(f"\n全部完成。汇总结果见: {summary_csv}")


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hyperparams import (
    DATASETS,
    DEFAULTS,
    MODELS,
    MODES,
    PARAM_LRS,
    PARAM_NUM_LAYERS,
)
from train import run_experiment
from utils import append_csv, get_device, load_dataset, save_json, set_seed


def parse_args():
    d = DEFAULTS
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=["cora", "citeseer", "flickr"],
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=["gcn", "gat", "graphsage", "gin"],
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=MODES,
        choices=["full", "sample"],
    )
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--flickr-epochs", type=int, default=d["flickr_epochs"])
    p.add_argument("--hidden", type=int, default=d["hidden"])
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


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = results_dir / "summary.csv"

    print("\n" + "#" * 60)
    print("# Part A: 模型 / 数据集 / full vs sample")
    print("#" * 60)

    for ds_name in args.datasets:
        data, num_features, num_classes = load_dataset(ds_name)
        epochs = args.flickr_epochs if ds_name == "flickr" else args.epochs

        for model_name in args.models:
            for mode in args.modes:
                tag = f"{ds_name}_{model_name}_{mode}_main"

                set_seed(args.seed)
                result = run_experiment(
                    dataset_name=ds_name,
                    data=data,
                    num_features=num_features,
                    num_classes=num_classes,
                    model_name=model_name,
                    mode=mode,
                    hidden_dim=args.hidden,
                    num_layers=args.num_layers,
                    heads=args.heads,
                    dropout=args.dropout,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    num_neighbors=None,
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
        print("# Part B: 学习率 / 层数 敏感性（Cora + GCN, full）")
        print("#" * 60)

        data, num_features, num_classes = load_dataset("cora")

        for lr in PARAM_LRS:
            for num_layers in PARAM_NUM_LAYERS:
                tag = f"cora_gcn_full_param_L{num_layers}_lr{lr}"

                set_seed(args.seed)
                result = run_experiment(
                    dataset_name="cora",
                    data=data,
                    num_features=num_features,
                    num_classes=num_classes,
                    model_name="gcn",
                    mode="full",
                    hidden_dim=args.hidden,
                    num_layers=num_layers,
                    heads=args.heads,
                    dropout=args.dropout,
                    lr=lr,
                    weight_decay=args.weight_decay,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    num_neighbors=None,
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

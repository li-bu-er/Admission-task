import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import run_experiment
from hyperparams import DEFAULTS
from utils import append_csv, get_device, load_dataset, save_json, set_seed


def parse_args():
    d = DEFAULTS
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        default="cora",
        choices=["cora", "citeseer", "flickr"],
    )
    p.add_argument(
        "--model",
        type=str,
        default="gcn",
        choices=["gcn", "gat", "sage", "graphsage", "gin"],
    )
    p.add_argument("--mode", type=str, default="full", choices=["full", "sample"])
    p.add_argument("--hidden", type=int, default=d["hidden"])
    p.add_argument("--num-layers", type=int, default=d["num_layers"])
    p.add_argument("--dropout", type=float, default=d["dropout"])
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--num-neighbors", type=int, nargs="+", default=None)
    p.add_argument("--heads", type=int, default=d["heads"])
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data, num_features, num_classes = load_dataset(name=args.dataset)
    device = get_device(prefer_cuda=not args.cpu)

    num_neighbors = args.num_neighbors
    if args.mode == "sample" and num_neighbors is None:
        num_neighbors = [15] * args.num_layers

    result = run_experiment(
        dataset_name=args.dataset,
        data=data,
        num_features=num_features,
        num_classes=num_classes,
        model_name=args.model,
        mode=args.mode,
        hidden_dim=args.hidden,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_neighbors=num_neighbors,
        device=device,
        verbose=True,
    )

    save_dir = (
        Path(args.save_dir)
        if args.save_dir
        else Path(__file__).resolve().parent.parent / "results"
    )
    tag = f"{args.dataset}_{args.model}_{args.mode}_L{args.num_layers}_lr{args.lr}"
    json_path = save_dir / f"{tag}.json"
    save_json(result, json_path)

    summary = {k: v for k, v in result.items() if k != "history"}
    append_csv(summary, save_dir / "summary.csv")
    print(f"结果已保存: {json_path}")
    print(f"汇总表追加: {save_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import run_experiment
from hyperparams import DEFAULTS
from utils import append_csv, get_device, load_graph_dataset, save_json, set_seed


def parse_args():
    d = DEFAULTS
    p = argparse.ArgumentParser(description="图分类单次实验")
    p.add_argument(
        "--dataset",
        type=str,
        default="mutag",
        help="TUDataset 名称（如 mutag/proteins）或 zinc",
    )
    p.add_argument(
        "--model",
        type=str,
        default="gcn",
        choices=["gcn", "gat", "sage", "graphsage", "gin"],
    )
    p.add_argument(
        "--pool",
        type=str,
        default="mean",
        choices=["mean", "avg", "max", "min"],
    )
    p.add_argument("--hidden", type=int, default=d["hidden"])
    p.add_argument("--embed-dim", type=int, default=d["embed_dim"])
    p.add_argument("--num-layers", type=int, default=d["num_layers"])
    p.add_argument("--dropout", type=float, default=d["dropout"])
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--heads", type=int, default=d["heads"])
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    train_set, val_set, test_set, in_dim, num_classes, task, display_name = (
        load_graph_dataset(args.dataset, seed=args.seed)
    )
    device = get_device(prefer_cuda=not args.cpu)

    result = run_experiment(
        dataset_name=display_name,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        task=task,
        model_name=args.model,
        pool=args.pool,
        in_dim=in_dim,
        hidden_dim=args.hidden,
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        verbose=True,
    )

    save_dir = (
        Path(args.save_dir)
        if args.save_dir
        else Path(__file__).resolve().parent.parent / "results"
    )
    ds_tag = args.dataset.lower().replace("-", "")
    pool_tag = "mean" if args.pool == "avg" else args.pool
    tag = f"{ds_tag}_{args.model}_{pool_tag}_L{args.num_layers}_lr{args.lr}"
    json_path = save_dir / f"{tag}.json"
    save_json(result, json_path)

    summary = {k: v for k, v in result.items() if k != "history"}
    append_csv(summary, save_dir / "summary.csv")
    print(f"结果已保存: {json_path}")
    print(f"汇总表追加: {save_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

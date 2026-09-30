import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import run_experiment
from hyperparams import DEFAULTS
from utils import append_csv, get_device, load_kg_dataset, save_json, set_seed


def parse_args():
    d = DEFAULTS
    p = argparse.ArgumentParser(description="知识图谱补全单次实验")
    p.add_argument(
        "--dataset",
        type=str,
        default="fb15k-237",
        choices=["fb15k-237", "wn18rr"],
    )
    p.add_argument(
        "--model",
        type=str,
        default="transe",
        choices=["transe", "rotate", "conve"],
    )
    p.add_argument("--embed-dim", type=int, default=d["embed_dim"])
    p.add_argument("--margin", type=float, default=d["margin"])
    p.add_argument("--dropout", type=float, default=d["dropout"])
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--num-neg", type=int, default=d["num_neg"])
    p.add_argument("--eval-every", type=int, default=d["eval_every"])
    p.add_argument("--eval-max-triples", type=int, default=None)
    p.add_argument("--seed", type=int, default=d["seed"])
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    kg_data = load_kg_dataset(args.dataset)
    device = get_device(prefer_cuda=not args.cpu)

    result = run_experiment(
        dataset_name=kg_data.name,
        kg_data=kg_data,
        model_name=args.model,
        embed_dim=args.embed_dim,
        margin=args.margin,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_neg=args.num_neg,
        eval_every=args.eval_every,
        eval_max_triples=args.eval_max_triples,
        device=device,
        verbose=True,
    )

    save_dir = (
        Path(args.save_dir)
        if args.save_dir
        else Path(__file__).resolve().parent.parent / "results"
    )
    tag = f"{args.dataset.replace('-', '')}_{args.model}_d{args.embed_dim}_lr{args.lr}"
    json_path = save_dir / f"{tag}.json"
    save_json(result, json_path)

    summary = {k: v for k, v in result.items() if k != "history"}
    append_csv(summary, save_dir / "summary.csv")
    print(f"结果已保存: {json_path}")
    print(f"汇总表追加: {save_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hyperparams import DATASETS, DEFAULTS, MODELS, PARAM_EMBED_DIMS, PARAM_LRS
from train import run_experiment
from utils import append_csv, get_device, load_kg_dataset, save_json, set_seed


def parse_args():
    d = DEFAULTS
    p = argparse.ArgumentParser(description="任务四知识图谱补全批量实验")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=["fb15k-237", "wn18rr"],
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=["transe", "rotate", "conve"],
    )
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--embed-dim", type=int, default=d["embed_dim"])
    p.add_argument("--margin", type=float, default=d["margin"])
    p.add_argument("--dropout", type=float, default=d["dropout"])
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--num-neg", type=int, default=d["num_neg"])
    p.add_argument("--eval-every", type=int, default=d["eval_every"])
    p.add_argument("--eval-max-triples", type=int, default=None)
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
    print("# Part A: 数据集 / 模型")
    print("#" * 60)

    for ds_name in args.datasets:
        kg_data = load_kg_dataset(ds_name)

        for model_name in args.models:
            tag = f"{ds_name.replace('-', '')}_{model_name}_main"

            set_seed(args.seed)
            result = run_experiment(
                dataset_name=kg_data.name,
                kg_data=kg_data,
                model_name=model_name,
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
            save_json(result, results_dir / f"{tag}.json")
            append_csv(
                {k: v for k, v in result.items() if k != "history"},
                summary_csv,
            )

    if not args.skip_param:
        print("\n" + "#" * 60)
        print("# Part B: 学习率 / 嵌入维度 敏感性（FB15k-237 + TransE）")
        print("#" * 60)

        kg_data = load_kg_dataset("fb15k-237")

        for lr in PARAM_LRS:
            for embed_dim in PARAM_EMBED_DIMS:
                tag = f"fb15237_transe_param_d{embed_dim}_lr{lr}"

                set_seed(args.seed)
                result = run_experiment(
                    dataset_name=kg_data.name,
                    kg_data=kg_data,
                    model_name="transe",
                    embed_dim=embed_dim,
                    margin=args.margin,
                    dropout=args.dropout,
                    lr=lr,
                    weight_decay=args.weight_decay,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    num_neg=args.num_neg,
                    eval_every=args.eval_every,
                    eval_max_triples=args.eval_max_triples,
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

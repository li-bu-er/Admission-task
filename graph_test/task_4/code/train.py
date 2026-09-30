import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import build_kg_model
from utils import get_device, timer


def corrupt_triples(triples, num_entities, num_neg=1):
    """随机替换头或尾实体，生成负样本。"""
    pos_h, pos_r, pos_t = triples[:, 0], triples[:, 1], triples[:, 2]
    batch_size = triples.size(0)
    neg_h, neg_r, neg_t = pos_h.clone(), pos_r.clone(), pos_t.clone()

    for i in range(batch_size):
        for _ in range(num_neg):
            if random.random() < 0.5:
                neg_h[i] = random.randint(0, num_entities - 1)
            else:
                neg_t[i] = random.randint(0, num_entities - 1)

    neg = torch.stack([neg_h, neg_r, neg_t], dim=1)
    return neg


def margin_loss(pos_score, neg_score, margin):
    return F.relu(margin - pos_score + neg_score).mean()


def train_one_epoch(model, triples, optimizer, device, num_entities, batch_size, margin, num_neg):
    model.train()
    perm = torch.randperm(triples.size(0))
    total_loss = 0.0
    steps = 0

    for start in range(0, triples.size(0), batch_size):
        idx = perm[start : start + batch_size]
        batch = triples[idx].to(device)
        neg = corrupt_triples(batch, num_entities, num_neg=num_neg).to(device)

        optimizer.zero_grad()
        pos_score = model.score(batch[:, 0], batch[:, 1], batch[:, 2])
        neg_score = model.score(neg[:, 0], neg[:, 1], neg[:, 2])
        loss = margin_loss(pos_score, neg_score, margin)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


def rank_filtered(scores, target, filter_set):
    """分数越高越好；过滤掉其它已知正确答案。"""
    target_score = scores[target].item()
    if filter_set:
        for ent in filter_set:
            if ent != target:
                scores[ent] = float("-inf")
    rank = int((scores > target_score).sum().item()) + 1
    return rank


@torch.no_grad()
def evaluate(model, triples, kg_data, device, max_triples=None):
    model.eval()
    if max_triples is not None and triples.size(0) > max_triples:
        triples = triples[torch.randperm(triples.size(0))[:max_triples]]

    ranks = []
    for h, r, t in tqdm(triples.tolist(), desc="eval", leave=False):
        h, r, t = int(h), int(r), int(t)

        tail_scores = model.score_all_tails(
            torch.tensor([h], device=device), torch.tensor([r], device=device)
        ).squeeze(0)
        tail_filter = kg_data.hr_to_tails.get((h, r), set())
        ranks.append(rank_filtered(tail_scores.clone(), t, tail_filter))

        head_scores = model.score_all_heads(
            torch.tensor([r], device=device), torch.tensor([t], device=device)
        ).squeeze(0)
        head_filter = kg_data.rt_to_heads.get((r, t), set())
        ranks.append(rank_filtered(head_scores.clone(), h, head_filter))

    ranks = torch.tensor(ranks, dtype=torch.float)
    mrr = (1.0 / ranks).mean().item()
    hits1 = (ranks <= 1).float().mean().item()
    hits3 = (ranks <= 3).float().mean().item()
    hits10 = (ranks <= 10).float().mean().item()
    return mrr, hits1, hits3, hits10


def run_experiment(
    dataset_name,
    kg_data,
    model_name,
    embed_dim,
    margin,
    dropout,
    lr,
    weight_decay,
    epochs,
    batch_size,
    num_neg,
    eval_every,
    eval_max_triples,
    device,
    verbose,
):
    device = device or get_device()
    model = build_kg_model(
        name=model_name,
        num_entities=kg_data.num_entities,
        num_relations=kg_data.num_relations,
        embed_dim=embed_dim,
        margin=margin,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_triples = kg_data.train

    best_mrr = 0.0
    best_metrics = {}
    best_epoch = 0
    history = []

    with timer() as t:
        epoch_iter = range(1, epochs + 1)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc=f"{model_name}")
        for epoch in epoch_iter:
            loss = train_one_epoch(
                model,
                train_triples,
                optimizer,
                device,
                kg_data.num_entities,
                batch_size,
                margin,
                num_neg,
            )

            record = {"epoch": epoch, "loss": loss}
            if epoch % eval_every == 0 or epoch == epochs:
                val_mrr, v1, v3, v10 = evaluate(
                    model, kg_data.valid, kg_data, device, eval_max_triples
                )
                test_mrr, t1, t3, t10 = evaluate(
                    model, kg_data.test, kg_data, device, eval_max_triples
                )
                record.update(
                    {
                        "val_mrr": val_mrr,
                        "val_hits1": v1,
                        "val_hits3": v3,
                        "val_hits10": v10,
                        "test_mrr": test_mrr,
                        "test_hits1": t1,
                        "test_hits3": t3,
                        "test_hits10": t10,
                    }
                )
                if val_mrr >= best_mrr:
                    best_mrr = val_mrr
                    best_epoch = epoch
                    best_metrics = {
                        "best_val_mrr": val_mrr,
                        "best_val_hits1": v1,
                        "best_val_hits3": v3,
                        "best_val_hits10": v10,
                        "best_test_mrr": test_mrr,
                        "best_test_hits1": t1,
                        "best_test_hits3": t3,
                        "best_test_hits10": t10,
                    }

            history.append(record)

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "embed_dim": embed_dim,
        "margin": margin,
        "dropout": dropout,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_neg": num_neg,
        "best_epoch": best_epoch,
        "train_time_sec": t["elapsed"],
        "device": str(device),
        **best_metrics,
    }
    if verbose:
        print(
            f"[{dataset_name}|{model_name}] best_val_mrr={best_mrr:.4f} "
            f"test_mrr@{best_epoch}={best_metrics.get('best_test_mrr', 0):.4f} "
            f"hits@10={best_metrics.get('best_test_hits10', 0):.4f} "
            f"time={t['elapsed']:.2f}s"
        )

    result["history"] = history
    return result

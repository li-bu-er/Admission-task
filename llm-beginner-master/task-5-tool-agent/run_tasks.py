"""在 data/tasks.json 上跑 ReActAgent，打印每题 trace 与关键词命中情况。"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agent import ReActAgent  # noqa: E402


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = text.replace(",", "").replace("，", "")
    return re.sub(r"\s+", "", text)


def answer_matches(answer: str, expected_keywords) -> bool:
    norm_answer = normalize_answer(answer)
    for expected in expected_keywords:
        if isinstance(expected, list):
            if not any(normalize_answer(k) in norm_answer for k in expected):
                return False
        elif normalize_answer(expected) not in norm_answer:
            return False
    return True


def main(args: argparse.Namespace) -> None:
    tasks_path = ROOT / "data" / "tasks.json"
    if not tasks_path.exists():
        raise SystemExit("缺少 data/tasks.json，请先 python data/download.py")

    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if args.id is not None:
        tasks = [t for t in tasks if t["id"] == args.id]
        if not tasks:
            raise SystemExit(f"找不到 id={args.id}")

    agent = ReActAgent(inject_error=args.inject_error, max_steps=args.max_steps)
    success = 0
    reports = []

    for t in tasks:
        print("\n" + "=" * 60)
        print(f"[{t['id']}] {t['task']}")
        trace = agent.run(t["task"])
        final = trace.get("final_answer", "")
        ok = answer_matches(final, t.get("expected_answer_contains", []))
        success += int(ok)
        print(f"Final Answer: {final}")
        print(f"HIT={ok}")
        for i, step in enumerate(trace.get("steps", []), 1):
            print(f"\n-- step {i} --")
            if step.get("thought"):
                print(f"Thought: {step['thought']}")
            if step.get("action"):
                print(f"Action: {step['action']}")
                print(f"Action Input: {step.get('action_input')}")
            if step.get("observation") is not None:
                obs = str(step["observation"])
                print(f"Observation: {obs[:800]}")
        reports.append(
            {
                "id": t["id"],
                "success": ok,
                "final_answer": final,
                "steps": trace.get("steps", []),
            }
        )

    rate = success / max(len(tasks), 1)
    print(f"\n=== success rate: {success}/{len(tasks)} = {rate:.3f} ===")
    out = ROOT / "eval" / "traces.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"traces 已写入 {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, default=None, help="只跑某一题")
    p.add_argument("--max_steps", type=int, default=8)
    p.add_argument("--inject_error", action="store_true", help="首次工具调用注入错误")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

"""在 data/toy-repo 上运行 CodingAgent，打印 Trace。"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agent import CodingAgent  # noqa: E402


def main(args: argparse.Namespace) -> None:
    toy = ROOT / "data" / "toy-repo"
    issue_path = toy / "ISSUE.md"
    if not issue_path.exists():
        raise SystemExit("缺少 data/toy-repo，请先 python data/download.py")

    # 与自检一致：从 buggy 快照恢复
    orig = toy / "calculator.py.orig"
    if orig.exists():
        shutil.copy(orig, toy / "calculator.py")

    agent = CodingAgent(use_llm=not args.no_llm, max_steps=args.max_steps)
    issue = issue_path.read_text(encoding="utf-8")
    trace = agent.run(repo_path=str(toy), issue=issue)

    print("tests_passed =", trace.get("tests_passed"))
    print("steps =", len(trace.get("steps") or []))
    print("\n===== patch (preview) =====")
    print(str(trace.get("patch", ""))[:1200])
    print("\n===== steps =====")
    for i, s in enumerate(trace.get("steps") or [], 1):
        tc = s.get("tool_call")
        name = tc.get("name") if isinstance(tc, dict) else None
        print(f"[{i}] thought={s.get('thought')!r} tool={name}")
        obs = str(s.get("observation", ""))
        print(f"    obs={obs[:200]!r}")

    out = ROOT / "eval" / "toy_trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ntrace 已写入 {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--no-llm", action="store_true", help="仅跑确定性工具路径（toy-repo）")
    p.add_argument("--max_steps", type=int, default=12)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

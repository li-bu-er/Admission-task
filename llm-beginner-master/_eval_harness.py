"""各任务 eval/run.py 共用的自检运行壳。

test_* 返回含 test/pass 的 dict（True/None/False），这里统一打印并写 result.json。
"""
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

_TAGS = {True: "[通过]", None: "[跳过]", False: "[失败]"}


def setup_console() -> None:
    # Windows 控制台默认 CP936 会乱码；result.json 仍是 UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_tests(tests: list[Callable[[], dict]], root: Path) -> list[dict]:
    setup_console()
    results: list[dict] = []
    for fn in tests:
        try:
            r = fn()
        except Exception as e:
            r = {"test": fn.__name__.removeprefix("test_"),
                 "pass": False, "error": str(e),
                 "trace": traceback.format_exc().splitlines()[-3:]}
        results.append(r)
        tag = _TAGS.get(r.get("pass"), "[失败]")
        print(f"{tag} {r['test']}: {json.dumps(r, ensure_ascii=False)}")

    out = root / "eval" / "result.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out.relative_to(root)}")
    return results

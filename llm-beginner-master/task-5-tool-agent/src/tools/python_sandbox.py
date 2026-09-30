"""受限 Python exec（超时 + import 白名单 + 抓 stdout）。

⚠️ 非真正隔离，勿对不可信输入使用。
"""
from __future__ import annotations

import builtins
import io
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from contextlib import redirect_stdout
from typing import Any, Dict

TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": (
            "在受限 Python 环境中执行短代码并返回 stdout。"
            "适合写小函数、循环、打印结果；禁止危险 import。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码，请用 print 输出结果",
                }
            },
            "required": ["code"],
        },
    },
}

_BLOCKED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "importlib",
    "ctypes",
    "multiprocessing",
    "threading",
    "pickle",
    "requests",
    "urllib",
    "http",
}

_ALLOWED_BUILTINS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "filter",
    "float",
    "int",
    "len",
    "list",
    "map",
    "max",
    "min",
    "print",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "True",
    "False",
    "None",
    "Exception",
    "ValueError",
    "TypeError",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root in _BLOCKED_MODULES:
        raise ImportError(f"禁止 import {name}")
    # 只放行少量纯计算库
    if root not in {"math", "statistics", "json", "re", "collections", "itertools", "functools"}:
        raise ImportError(f"不允许 import {name}（教学沙箱白名单外）")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _make_globals() -> dict:
    safe_builtins = {k: getattr(builtins, k) for k in _ALLOWED_BUILTINS if hasattr(builtins, k)}
    safe_builtins["__import__"] = _safe_import
    return {"__builtins__": safe_builtins}


def _exec_code(code: str) -> str:
    buf = io.StringIO()
    g = _make_globals()
    with redirect_stdout(buf):
        exec(code, g, g)  # noqa: S102  # 教学沙箱有意为之
    out = buf.getvalue()
    if out.strip():
        return out

    # 模型常忘 print：无 stdout 时尝试 eval 最后一行表达式
    lines = [ln for ln in code.strip().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return out
    last = lines[-1].strip()
    if last.startswith(("import ", "from ", "def ", "class ", "for ", "while ", "if ", "elif ", "else", "try", "with ", "return", "pass")):
        return out
    try:
        with redirect_stdout(buf):
            value = eval(last, g, g)  # noqa: S307  # 教学沙箱：仅在无 stdout 时兜底
            if value is not None:
                print(repr(value) if not isinstance(value, str) else value)
        return buf.getvalue()
    except Exception:
        return out


def run(args: dict, timeout: float = 3.0) -> str:
    code = str(args["code"])
    if not code.strip():
        raise ValueError("code 为空")

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_exec_code, code)
        try:
            out = fut.result(timeout=timeout)
        except FuturesTimeout as e:
            raise TimeoutError(f"代码执行超过 {timeout}s") from e
        except Exception as e:
            raise RuntimeError(f"执行失败: {e}\n{traceback.format_exc(limit=3)}") from e

    out = out if out.strip() else "(无 stdout 输出；请用 print(...) 打印结果)"
    return out

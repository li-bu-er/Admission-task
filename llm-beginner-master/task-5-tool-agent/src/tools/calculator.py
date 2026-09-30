"""安全四则运算 + 常用数学函数（禁止任意 eval）。"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any, Dict

TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "计算数学表达式。支持四则运算、幂、括号，以及 "
            "sqrt/sin/cos/tan/log/ln/abs/floor/ceil/pi/e 等。"
            "输入如 'sqrt(2026)' 或 '(123+456)*789'。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式字符串",
                }
            },
            "required": ["expression"],
        },
    },
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log10,
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "exp": math.exp,
    "pow": math.pow,
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    # Python <3.10 还有 ast.Num
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return float(node.n)  # type: ignore[attr-defined]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _FUNCS:
        val = _FUNCS[node.id]
        if callable(val):
            raise ValueError(f"函数 {node.id} 需要括号调用，如 {node.id}(x)")
        return float(val)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            fn = _FUNCS[node.func.id]
            if not callable(fn):
                raise ValueError(f"{node.func.id} 是常量不是函数")
            args = [_eval_node(a) for a in node.args]
            return float(fn(*args))
        raise ValueError("只允许白名单数学函数")
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


def run(args: dict) -> str:
    expr = str(args["expression"]).strip()
    if not expr:
        raise ValueError("expression 为空")
    # ^ → **；全角括号
    expr = expr.replace("^", "**").replace("（", "(").replace("）", ")")
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_node(tree)
    except Exception as e:
        raise ValueError(f"无法计算表达式 {expr!r}: {e}") from e
    # 整数去 .0；否则固定 6 位（如 sqrt(2026)→45.011110）
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:.6f}"

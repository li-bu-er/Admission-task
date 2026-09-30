"""本地文件检索：文件名/内容匹配；带路径越界保护。"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Dict, List

TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": (
            "在指定目录下按文件名或文件内容搜索。"
            "pattern 可为文件名片段、扩展名（如 .md）或正文关键词（如 TODO）。"
            "返回匹配路径及内容片段。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "搜索模式：文件名或内容关键词",
                },
                "dir": {
                    "type": "string",
                    "description": "要搜索的根目录路径",
                },
            },
            "required": ["pattern", "dir"],
        },
    },
}

# 禁止跳出任务根目录
TASK_ROOT = Path(__file__).resolve().parents[2]


def _ensure_under_root(path: Path, allowed_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(allowed_root.resolve())
    except ValueError as e:
        raise PermissionError(
            f"路径越界：{resolved} 不在允许根目录 {allowed_root} 内"
        ) from e
    return resolved


def run(args: dict, max_hits: int = 20, snippet_chars: int = 400) -> str:
    pattern = str(args["pattern"])
    dir_arg = str(args["dir"])
    root = Path(dir_arg).expanduser()
    if not root.is_absolute():
        root = (TASK_ROOT / root).resolve()
    else:
        root = root.resolve()

    allowed = TASK_ROOT.resolve()
    root = _ensure_under_root(root, allowed)

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")

    pat_lower = pattern.lower()
    hits: List[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        try:
            rel = path.relative_to(TASK_ROOT)
        except ValueError:
            rel = path

        # *.md / .md 按文件名；普通关键词也可扫正文
        if any(ch in pattern for ch in "*?[]"):
            name_match = fnmatch.fnmatch(path.name.lower(), pat_lower)
        elif pattern.startswith(".") and "/" not in pattern and "\\" not in pattern:
            name_match = path.suffix.lower() == pat_lower
        else:
            name_match = pat_lower in path.name.lower()
        content_match = False
        snippet = ""
        # glob / 纯扩展名只按文件名，避免把 "*.md" 当正文关键词
        allow_content = not (
            any(ch in pattern for ch in "*?[]")
            or (pattern.startswith(".") and "/" not in pattern and "\\" not in pattern)
        )
        try:
            if path.stat().st_size <= 512_000 and path.suffix.lower() in {
                ".md",
                ".txt",
                ".py",
                ".json",
                ".jsonl",
                ".csv",
                ".tex",
                ".yml",
                ".yaml",
                "",
            }:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if allow_content and pat_lower in text.lower():
                    content_match = True
                    idx = text.lower().find(pat_lower)
                    if idx < 0:
                        snippet = text[:snippet_chars]
                    else:
                        lo = max(0, idx - 40)
                        snippet = text[lo : lo + snippet_chars]
        except OSError:
            continue

        if name_match or content_match:
            block = [f"- path: {rel.as_posix()}"]
            if snippet:
                block.append(f"  snippet: {snippet!r}")
            elif name_match:
                # 按名命中也附开头（读 README 第一段需要 content_head）
                try:
                    head = path.read_text(encoding="utf-8", errors="ignore")[:snippet_chars]
                    if head.strip():
                        block.append(f"  content_head: {head!r}")
                except OSError:
                    pass
            hits.append("\n".join(block))
        if len(hits) >= max_hits:
            break

    if not hits:
        return f"在 {root} 下未找到匹配 {pattern!r} 的文件。"
    return f"共 {len(hits)} 个匹配：\n" + "\n".join(hits)

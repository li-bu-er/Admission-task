"""仓库内原子工具；MCP / CodingAgent 共用。

路径 resolve 后必须落在 repo 内；subprocess 用 list、禁止 shell=True。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class RepoTools:
    def __init__(self, repo_path: str | Path):
        self.repo = Path(repo_path).resolve()
        if not self.repo.is_dir():
            raise FileNotFoundError(f"repo 不存在: {self.repo}")

    def _safe_path(self, rel: str) -> Path:
        # 相对路径规范到 repo 内；拒绝越界 / 绝对路径逃逸
        raw = Path(rel)
        # 禁止以绝对路径直接指向 repo 外
        candidate = (self.repo / rel).resolve() if not raw.is_absolute() else raw.resolve()
        try:
            candidate.relative_to(self.repo)
        except ValueError as e:
            raise PermissionError(f"路径越界拒绝: {rel} -> {candidate}") from e
        return candidate

    def read_file(self, path: str) -> str:
        p = self._safe_path(path)
        if not p.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")
        return p.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> str:
        p = self._safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="\n")
        return f"OK wrote {path} ({len(content)} chars)"

    def list_dir(self, path: str = ".") -> str:
        p = self._safe_path(path)
        if not p.is_dir():
            raise NotADirectoryError(path)
        entries = []
        for child in sorted(p.iterdir())[:200]:
            tag = "dir" if child.is_dir() else "file"
            entries.append(f"{tag}\t{child.relative_to(self.repo).as_posix()}")
        return "\n".join(entries) if entries else "(empty)"

    def run_tests(self, args: Optional[List[str]] = None) -> Dict[str, Any]:
        cmd = [sys.executable, "-m", "pytest", "-q"]
        if args:
            cmd.extend(args)
        proc = subprocess.run(
            cmd,
            cwd=str(self.repo),
            text=True,
            capture_output=True,
            timeout=120,
            shell=False,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return {
            "passed": proc.returncode == 0,
            "returncode": proc.returncode,
            "output": output[-4000:],
        }

    def git_diff(self) -> str:
        if (self.repo / ".git").exists():
            proc = subprocess.run(
                ["git", "diff", "--", "."],
                cwd=str(self.repo),
                text=True,
                capture_output=True,
                timeout=30,
                shell=False,
            )
            diff = proc.stdout or ""
            if diff.strip():
                return diff
        # 无 git 时退化为跟 *.orig 对比
        orig = self.repo / "calculator.py.orig"
        cur = self.repo / "calculator.py"
        if orig.exists() and cur.exists():
            if orig.read_text(encoding="utf-8") != cur.read_text(encoding="utf-8"):
                return (
                    "--- a/calculator.py\n+++ b/calculator.py\n"
                    f"-{orig.read_text(encoding='utf-8')!r}\n"
                    f"+{cur.read_text(encoding='utf-8')!r}\n"
                )
        return ""

    def git_apply(self, patch: str) -> str:
        if not patch.strip():
            raise ValueError("patch 为空")
        tmp = self.repo / ".agent_patch.diff"
        try:
            tmp.write_text(patch, encoding="utf-8")
            if not (self.repo / ".git").exists():
                raise RuntimeError("当前 repo 无 .git，无法 git apply；请改用 write_file")
            proc = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", str(tmp.name)],
                cwd=str(self.repo),
                text=True,
                capture_output=True,
                timeout=30,
                shell=False,
            )
            if proc.returncode != 0:
                raise RuntimeError((proc.stdout or "") + (proc.stderr or ""))
            return "OK patch applied"
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def search_text(self, pattern: str, glob: str = "*.py") -> str:
        hits: List[str] = []
        for p in self.repo.rglob(glob):
            if not p.is_file():
                continue
            if ".git" in p.parts:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if pattern in line:
                    rel = p.relative_to(self.repo).as_posix()
                    hits.append(f"{rel}:{i}: {line.strip()}")
                    if len(hits) >= 50:
                        return "\n".join(hits)
        return "\n".join(hits) if hits else f"(no hit for {pattern!r})"


# MCP / Agent 用的工具元数据（≥5）
TOOL_DEFS: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "读取仓库内相对路径文件的全文",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "写入仓库内相对路径文件（覆盖）。不要修改测试文件。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "列出仓库内某目录的文件与子目录",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
            "required": [],
        },
    },
    {
        "name": "run_tests",
        "description": "在仓库根目录运行 python -m pytest -q，返回是否通过与输出",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "git_diff",
        "description": "查看当前工作区相对 HEAD（或 .orig）的 diff",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "git_apply",
        "description": "将 unified diff 应用到当前仓库（勿使用 reset --hard 等破坏性命令）",
        "input_schema": {
            "type": "object",
            "properties": {"patch": {"type": "string"}},
            "required": ["patch"],
        },
    },
    {
        "name": "search_text",
        "description": "在仓库源码中搜索包含某子串的行",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "glob": {"type": "string", "default": "*.py"},
            },
            "required": ["pattern"],
        },
    },
]


def dispatch_tool(tools: RepoTools, name: str, arguments: dict) -> str:
    # 异常转字符串，别打崩 server/agent
    try:
        if name == "read_file":
            return tools.read_file(arguments["path"])
        if name == "write_file":
            return tools.write_file(arguments["path"], arguments["content"])
        if name == "list_dir":
            return tools.list_dir(arguments.get("path", "."))
        if name == "run_tests":
            r = tools.run_tests()
            status = "PASSED" if r["passed"] else "FAILED"
            return f"{status}\n{r['output']}"
        if name == "git_diff":
            return tools.git_diff() or "(no diff)"
        if name == "git_apply":
            return tools.git_apply(arguments["patch"])
        if name == "search_text":
            return tools.search_text(arguments["pattern"], arguments.get("glob", "*.py"))
        return f"ERROR: unknown tool {name}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

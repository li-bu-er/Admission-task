"""Subagents：独立 context / 工具子集，主 agent 只收摘要。"""
from __future__ import annotations

from typing import Any, Dict, List

from ..repo_tools import RepoTools, dispatch_tool


class BaseSubagent:
    name: str = "base"
    max_steps: int = 4
    tool_allowlist: List[str] = []

    def __init__(self, tools: RepoTools):
        self.tools = tools
        self.steps: List[Dict[str, Any]] = []  # 独立 context

    def _call(self, tool_name: str, args: dict) -> str:
        if self.tool_allowlist and tool_name not in self.tool_allowlist:
            return f"ERROR: subagent {self.name} 不允许调用 {tool_name}"
        obs = dispatch_tool(self.tools, tool_name, args)
        self.steps.append(
            {"thought": f"[{self.name}] call {tool_name}", "tool_call": {"name": tool_name, "arguments": args}, "observation": obs}
        )
        return obs

    def run(self, task: str) -> str:
        raise NotImplementedError

    def summary(self) -> str:
        # 主 agent 只看摘要，别吞整段 observation
        if not self.steps:
            return f"[{self.name}] no steps"
        last = self.steps[-1]
        obs = str(last.get("observation", ""))
        if len(obs) > 400:
            obs = obs[:400] + "…"
        return f"[{self.name}] steps={len(self.steps)}; last_tool={last.get('tool_call', {}).get('name')}; obs={obs}"


class CodeSearchSubagent(BaseSubagent):
    name = "code_search"
    max_steps = 5
    tool_allowlist = ["list_dir", "search_text", "read_file"]

    def run(self, task: str) -> str:
        # 从任务抽关键词搜（不掺主 agent messages）
        keywords = []
        for token in ["add", "def add", "calculator", "return a - b", "factorial"]:
            if token.lower() in task.lower() or token in ("add", "calculator"):
                keywords.append(token if token != "def add" else "def add")
        keywords = list(dict.fromkeys(keywords)) or ["def "]

        findings = []
        self._call("list_dir", {"path": "."})
        for kw in keywords[:3]:
            hit = self._call("search_text", {"pattern": kw, "glob": "*.py"})
            findings.append(hit)
            if "calculator.py" in hit:
                findings.append(self._call("read_file", {"path": "calculator.py"}))
                break
        # 还没读到 calculator 就直接读
        if not any("def add" in f for f in findings):
            try:
                findings.append(self._call("read_file", {"path": "calculator.py"}))
            except Exception:
                pass
        return self.summary() + "\n---\n" + "\n".join(findings)[:1500]


class TestExecutorSubagent(BaseSubagent):
    name = "test_executor"
    max_steps = 2
    tool_allowlist = ["run_tests"]

    def run(self, task: str = "") -> str:
        out = self._call("run_tests", {})
        passed = out.startswith("PASSED")
        return f"[{self.name}] tests_passed={passed}\n{out[:800]}"

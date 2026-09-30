"""手写 ReAct Agent：Thought / Action / Action Input / Observation 循环。"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from .tools import TOOL_MODULES, all_schemas, dispatch

# OpenAI 兼容本地推理（Ollama / vLLM / llama.cpp）
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "qwen2.5:7b-instruct")


def _tool_catalog_text() -> str:
    lines = []
    for schema in all_schemas():
        fn = schema["function"]
        params = fn.get("parameters", {}).get("properties", {})
        req = fn.get("parameters", {}).get("required", [])
        param_desc = ", ".join(
            f"{k}{'*' if k in req else ''}: {v.get('type', 'any')}" for k, v in params.items()
        )
        lines.append(f"- {fn['name']}({param_desc}): {fn['description']}")
    return "\n".join(lines)


SYSTEM_PROMPT = """你是一个严格遵循 ReAct 格式的工具调用助手。
你必须使用下列格式（每次只输出一轮 Thought/Action，等待 Observation；或直接 Final Answer）：

Thought: <简要推理下一步做什么>
Action: <工具名，必须是下列之一>
Action Input: <JSON 对象，键名必须与工具参数一致>
Observation: <由系统填入，你不要编造>

当你已经能回答用户时：
Thought: <总结>
Final Answer: <最终答案，包含关键数字/关键词>

可用工具：
{tools}

规则：
1. Action 只能是工具名：calculator / python_sandbox / file_search / wiki
2. Action Input 必须是单行 JSON，例如 {{"expression": "1+2"}} 或 {{"code": "print(1)"}}
3. 不要在同一轮输出 Observation；不要一次输出多个 Action
4. 数值计算优先用 calculator；需要循环/自定义逻辑用 python_sandbox
5. 查百科用 wiki；查本地文件用 file_search（dir 用相对路径如 data/agent-fixtures）
6. Final Answer 必须覆盖题目问到的全部要点（例如既要计算结果，也要报位数）
7. python_sandbox 的 code 里必须用 print(...) 输出最终结果；看到“无 stdout”就立刻改代码加 print 再调用一次
8. 找某类文件用 pattern=".md" 或 "*.md"；读某个文件内容用 pattern="README.md"（会返回 content_head），不要搜索“第一段”这类自然语言

Few-shot 示例：
用户: 计算 2 + 3 * 4
Thought: 需要计算表达式，使用 calculator。
Action: calculator
Action Input: {{"expression": "2 + 3 * 4"}}
Observation: 14
Thought: 已经得到结果。
Final Answer: 14

用户: 用 Python 计算 1+2
Thought: 用 python_sandbox，并 print 结果。
Action: python_sandbox
Action Input: {{"code": "print(1+2)"}}
Observation: 3
Thought: 得到结果。
Final Answer: 3

用户: 读取 data/agent-fixtures/README.md 的开头
Thought: 用 file_search 按文件名匹配 README.md。
Action: file_search
Action Input: {{"pattern": "README.md", "dir": "data/agent-fixtures"}}
"""


def _build_messages(task: str, history: str) -> List[Dict[str, str]]:
    system = SYSTEM_PROMPT.format(tools=_tool_catalog_text())
    user = f"任务：{task}\n\n到目前为止的轨迹：\n{history}\n\n请输出下一步（Thought + Action/Action Input，或 Final Answer）。"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_llm_output(text: str) -> Dict[str, Any]:
    text = text.strip()
    result: Dict[str, Any] = {
        "thought": None,
        "action": None,
        "action_input": None,
        "final_answer": None,
        "raw": text,
    }

    # 允许中英文冒号
    m_final = re.search(
        r"Final\s*Answer\s*[:：]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL
    )
    if m_final:
        result["final_answer"] = m_final.group(1).strip()

    m_thought = re.search(
        r"Thought\s*[:：]\s*(.+?)(?=\n\s*Action\s*[:：]|\n\s*Final\s*Answer\s*[:：]|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_thought:
        result["thought"] = m_thought.group(1).strip()

    m_action = re.search(
        r"Action\s*[:：]\s*([a-zA-Z_][a-zA-Z0-9_]*)",
        text,
        flags=re.IGNORECASE,
    )
    if m_action:
        result["action"] = m_action.group(1).strip()

    m_input = re.search(
        r"Action\s*Input\s*[:：]\s*(.+?)(?=\n\s*Observation\s*[:：]|\n\s*Thought\s*[:：]|\n\s*Final\s*Answer\s*[:：]|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_input:
        raw_in = m_input.group(1).strip()
        raw_in = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_in).strip()
        result["action_input"] = _parse_action_input(raw_in, result.get("action"))

    # 有 Action 时本轮先跑工具，不直接结束（即使顺带写了 Final Answer）
    if result["action"]:
        result["final_answer"] = None
    return result


def _parse_action_input(raw: str, action: Optional[str]) -> dict:
    # 容错非严格 JSON；按工具猜单字段
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    if action == "calculator":
        return {"expression": raw.strip("\"'")}
    if action == "python_sandbox":
        return {"code": raw}
    if action == "wiki":
        return {"query": raw.strip("\"'")}
    if action == "file_search":
        return {"pattern": raw.strip("\"'"), "dir": "data/agent-fixtures"}
    return {"raw": raw}


class ReActAgent:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        max_steps: int = 10,
        temperature: float = 0.2,
        inject_error: bool = False,
    ):
        # inject_error：首次工具调用注入错误 Observation（S4 钩子）
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_steps = max_steps
        self.temperature = temperature
        self.inject_error = inject_error
        self._error_injected = False
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _llm(self, messages: List[Dict[str, str]]) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=768,
        )
        return (resp.choices[0].message.content or "").strip()

    def _call_tool(self, name: str, args: dict) -> str:
        # 异常转成错误字符串，方便错误恢复
        if self.inject_error and not self._error_injected:
            self._error_injected = True
            return (
                "ERROR: injected failure for recovery test. "
                "请修正参数或换一种方式后重试。"
            )
        try:
            return str(dispatch(name, args))
        except Exception as e:
            return f"ERROR: tool `{name}` failed: {type(e).__name__}: {e}"

    def run(self, task: str) -> Dict[str, Any]:
        self._error_injected = False
        steps: List[Dict[str, Any]] = []
        history = "(尚无步骤)"
        final_answer = ""
        parse_fail_streak = 0

        for _ in range(self.max_steps):
            messages = _build_messages(task, history)
            try:
                raw = self._llm(messages)
            except Exception as e:
                return {
                    "steps": steps,
                    "final_answer": f"LLM 调用失败: {e}",
                    "success": False,
                }

            parsed = _parse_llm_output(raw)

            if parsed["final_answer"] and not parsed["action"]:
                final_answer = parsed["final_answer"]
                steps.append(
                    {
                        "thought": parsed.get("thought"),
                        "action": None,
                        "action_input": None,
                        "observation": None,
                        "tool": None,
                        "raw": raw,
                    }
                )
                break

            action = parsed.get("action")
            action_input = parsed.get("action_input") or {}

            if not action or action not in TOOL_MODULES:
                parse_fail_streak += 1
                obs = (
                    f"ERROR: 无法解析 Action（得到 {action!r}）。"
                    f"请严格使用格式 Action: <工具名> 与 Action Input: {{...}}。"
                    f"可用工具: {list(TOOL_MODULES)}"
                )
                step = {
                    "thought": parsed.get("thought"),
                    "action": action,
                    "action_input": action_input,
                    "observation": obs,
                    "tool": action,
                    "raw": raw,
                }
                steps.append(step)
                history = self._format_history(steps)
                if parse_fail_streak >= 3:
                    final_answer = "未能解析有效工具调用，任务中止。"
                    break
                continue

            parse_fail_streak = 0
            observation = self._call_tool(action, action_input)
            step = {
                "thought": parsed.get("thought"),
                "action": action,
                "action_input": action_input,
                "observation": observation,
                "tool": action,  # 自检可读 tool / action
                "raw": raw,
            }
            steps.append(step)
            history = self._format_history(steps)
        else:
            # 步数耗尽：用最后 observation 兜底
            if not final_answer and steps:
                last_obs = steps[-1].get("observation") or ""
                final_answer = f"（步数用尽，最后观察）{last_obs[:500]}"

        return {
            "steps": steps,
            "final_answer": final_answer,
            "success": bool(final_answer.strip()),
        }

    @staticmethod
    def _format_history(steps: List[Dict[str, Any]]) -> str:
        parts = []
        for i, s in enumerate(steps, 1):
            parts.append(f"--- step {i} ---")
            if s.get("thought"):
                parts.append(f"Thought: {s['thought']}")
            if s.get("action"):
                parts.append(f"Action: {s['action']}")
                parts.append(
                    f"Action Input: {json.dumps(s.get('action_input') or {}, ensure_ascii=False)}"
                )
            if s.get("observation") is not None:
                parts.append(f"Observation: {s['observation']}")
        return "\n".join(parts) if parts else "(尚无步骤)"

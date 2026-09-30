"""检索上下文 → Prompt → 本地 Qwen（OpenAI 兼容 API）。"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import requests

DEFAULT_API_BASE = os.environ.get("RAG_API_BASE", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("RAG_MODEL", "qwen2.5:7b-instruct")


def build_prompt(query: str, sources: List[Dict], max_chars: int = 3500) -> str:
    # 只能依据上下文答；去重 + 截断
    seen = set()
    blocks = []
    used = 0
    for i, s in enumerate(sources, 1):
        text = (s.get("text") or "").strip()
        # 去空白后去重，避免 overlap chunk 重复占额度
        key = "".join(text.split())
        if not key or key in seen:
            continue
        seen.add(key)
        src = s.get("source", f"doc-{i}")
        block = f"[{i}] 来源: {src}\n{text}"
        if used + len(block) > max_chars:
            remain = max_chars - used
            if remain > 100:
                blocks.append(block[:remain] + "…")
            break
        blocks.append(block)
        used += len(block)

    context = "\n\n".join(blocks) if blocks else "（无检索到相关片段）"
    return (
        "你是严谨的中文助教。请只根据「检索上下文」回答用户问题；"
        "若上下文不足以支持答案，请明确说「根据提供的资料无法确定」。"
        "不要编造上下文中没有的事实；回答简洁，必要时引用片段编号。\n\n"
        f"【检索上下文】\n{context}\n\n"
        f"【用户问题】\n{query}\n\n"
        "【回答】\n"
    )


def call_chat_api(
    prompt: str,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
) -> Optional[str]:
    # 失败返回 None，上层走抽取式兜底
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[generator] API 调用失败（{e}），将使用抽取式兜底")
        return None


def extractive_fallback(query: str, sources: List[Dict]) -> str:
    # 无 LLM 时保证端到端非空
    if not sources:
        return "根据提供的资料无法确定。"
    top = sources[0]["text"].strip().replace("\n", " ")
    if len(top) > 400:
        top = top[:400] + "…"
    return f"根据检索到的资料（摘录）：{top}"


def generate_answer(
    query: str,
    sources: List[Dict],
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
) -> str:
    prompt = build_prompt(query, sources)
    text = call_chat_api(prompt, api_base=api_base, model=model)
    if text and text.strip():
        return text.strip()
    return extractive_fallback(query, sources)

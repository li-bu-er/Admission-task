"""维基百科摘要（MediaWiki API，中英均可）。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import requests

TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": (
            "查询维基百科摘要。支持中文与英文条目名或关键词，"
            "如「图灵机」「Geoffrey Hinton」「Transformer (machine learning model)」。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "条目名或搜索关键词",
                }
            },
            "required": ["query"],
        },
    },
}

_UA = {"User-Agent": "llm-beginner-task5/1.0 (educational; contact: local)"}


def _api_get(lang: str, params: dict) -> dict:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    r = requests.get(url, params=params, headers=_UA, timeout=15)
    r.raise_for_status()
    return r.json()


def _search_title(lang: str, query: str) -> Optional[str]:
    data = _api_get(
        lang,
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 3,
            "format": "json",
        },
    )
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None
    return hits[0]["title"]


def _fetch_extract(lang: str, title: str) -> Tuple[str, str]:
    data = _api_get(
        lang,
        {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json",
            "redirects": 1,
        },
    )
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        extract = (page.get("extract") or "").strip()
        t = page.get("title") or title
        return t, extract
    return title, ""


def run(args: dict) -> str:
    query = str(args["query"]).strip()
    if not query:
        raise ValueError("query 为空")

    # 含中文优先中文站，否则先英后中
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in query)
    langs: List[str] = ["zh", "en"] if has_cjk else ["en", "zh"]

    errors = []
    for lang in langs:
        try:
            title = _search_title(lang, query) or query
            title, extract = _fetch_extract(lang, title)
            if extract:
                text = f"[{lang}.wikipedia] {title}\n{extract}"
                if len(text) > 2500:
                    text = text[:2500] + "…"
                return text
        except Exception as e:
            errors.append(f"{lang}: {e}")
            continue

    raise RuntimeError("维基百科查询失败：" + "; ".join(errors) if errors else "无结果")

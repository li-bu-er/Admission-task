"""MOSS SFT jsonl / DPO 偏好对加载。"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional

# MOSS 控制符，训练时剥掉只留正文
_MOSS_TAG_RE = re.compile(
    r"<\|(?:Human|MOSS|Inner Thoughts|Commands|Results)\|>:\s*"
    r"|</?(?:eoh|eom|eot|eoc|eor)>",
    flags=re.IGNORECASE,
)


def _norm_role(role: str) -> str:
    r = role.strip().lower()
    if r in {"human", "user", "prompter"}:
        return "user"
    if r in {"assistant", "gpt", "bot", "completion", "moss"}:
        return "assistant"
    if r in {"system"}:
        return "system"
    return r


def _strip_moss_tags(text: str) -> str:
    text = _MOSS_TAG_RE.sub("", text)
    return text.strip()


def _moss_chat_dict_to_messages(chat: dict, meta_instruction: Optional[str] = None):
    """moss-003：chat = {turn_1: {Human, MOSS}, ...}"""
    msgs: List[Dict[str, str]] = []
    if meta_instruction and str(meta_instruction).strip():
        sys_txt = str(meta_instruction).strip()
        if len(sys_txt) > 400:
            sys_txt = sys_txt[:400].rstrip() + "…"
        msgs.append({"role": "system", "content": sys_txt})

    def _turn_key(k: str):
        m = re.match(r"turn_(\d+)$", str(k), flags=re.IGNORECASE)
        return int(m.group(1)) if m else 10**9

    for key in sorted(chat.keys(), key=_turn_key):
        turn = chat[key]
        if not isinstance(turn, dict):
            continue
        human = turn.get("Human") or turn.get("human")
        moss = turn.get("MOSS") or turn.get("moss") or turn.get("Assistant")
        if human is not None:
            content = _strip_moss_tags(str(human))
            if content:
                msgs.append({"role": "user", "content": content})
        if moss is not None:
            content = _strip_moss_tags(str(moss))
            if content:
                msgs.append({"role": "assistant", "content": content})
    return msgs if any(m["role"] == "assistant" for m in msgs) else None


def _to_messages(obj: dict) -> Optional[List[Dict[str, str]]]:
    """统一成 [{role, content}, ...]。"""
    chat = obj.get("chat")
    # moss-003-sft-no-tools：chat 是 dict
    if isinstance(chat, dict):
        return _moss_chat_dict_to_messages(chat, obj.get("meta_instruction"))

    if "messages" in obj and isinstance(obj["messages"], list):
        raw = obj["messages"]
    elif isinstance(chat, list):
        raw = chat
    elif "conversation" in obj and isinstance(obj["conversation"], list):
        raw = obj["conversation"]
    elif "conversations" in obj and isinstance(obj["conversations"], list):
        raw = obj["conversations"]
    else:
        return None

    msgs: List[Dict[str, str]] = []
    for turn in raw:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("from") or turn.get("speaker")
        content = turn.get("content") or turn.get("text") or turn.get("value")
        if role is None or content is None:
            continue
        msgs.append({"role": _norm_role(str(role)), "content": str(content)})
    return msgs if msgs else None


def iter_jsonl(path: Path) -> Iterator[dict]:
    """逐行读 jsonl；zip 则读其中第一个 .jsonl。"""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if n.endswith(".jsonl")]
            if not names:
                raise FileNotFoundError(f"{path} 内无 jsonl")
            with zf.open(names[0]) as f:
                for line in f:
                    line = line.decode("utf-8").strip()
                    if line:
                        yield json.loads(line)
        return

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_sft_messages(
    data_path: str,
    max_samples: Optional[int] = 10000,
) -> List[List[Dict[str, str]]]:
    path = Path(data_path)
    if path.is_dir():
        cands = list(path.glob("*.jsonl")) + list(path.glob("*.zip"))
        if not cands:
            raise FileNotFoundError(f"{path} 下没有 jsonl/zip")
        path = cands[0]

    out: List[List[Dict[str, str]]] = []
    for obj in iter_jsonl(path):
        msgs = _to_messages(obj)
        if not msgs:
            continue
        roles = {m["role"] for m in msgs}
        if "assistant" not in roles:
            continue
        out.append(msgs)
        if max_samples is not None and len(out) >= max_samples:
            break
    return out


def _dpo_pair_from_obj(obj: dict) -> Optional[dict]:
    if "chosen" in obj and "rejected" in obj:
        prompt = obj.get("prompt") or obj.get("question") or obj.get("instruction")
        chosen = obj["chosen"]
        rejected = obj["rejected"]
        if isinstance(chosen, dict):
            chosen = chosen.get("content") or chosen.get("value") or chosen.get("text")
        if isinstance(rejected, dict):
            rejected = rejected.get("content") or rejected.get("value") or rejected.get("text")
        if prompt is None and "conversations" in obj:
            # ShareGPT：conversations 是历史，chosen/rejected 是最后回复
            conv = obj["conversations"]
            parts = []
            for t in conv:
                role = _norm_role(str(t.get("from", t.get("role", "user"))))
                val = t.get("value") or t.get("content") or ""
                parts.append({"role": role, "content": str(val)})
            return {
                "messages_context": parts,
                "chosen": str(chosen),
                "rejected": str(rejected),
            }
        if prompt is not None and chosen is not None and rejected is not None:
            return {
                "messages_context": [{"role": "user", "content": str(prompt)}],
                "chosen": str(chosen),
                "rejected": str(rejected),
            }
    return None


def load_dpo_pairs(
    data_path: str,
    max_samples: Optional[int] = 5000,
) -> List[dict]:
    """[{messages_context, chosen, rejected}, ...]"""
    path = Path(data_path)
    if path.is_dir():
        cands = (
            list(path.glob("*.jsonl"))
            + list(path.glob("*.json"))
            + list(path.glob("**/*.parquet"))
        )
        if not cands:
            raise FileNotFoundError(f"{path} 下没有可解析的 DPO 数据")
        path = cands[0]

    out: List[dict] = []
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            pair = _dpo_pair_from_obj(row.to_dict())
            if pair:
                out.append(pair)
            if max_samples is not None and len(out) >= max_samples:
                break
        return out

    if path.suffix == ".json" and not path.name.endswith(".jsonl"):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("data", [data])
        for obj in rows:
            pair = _dpo_pair_from_obj(obj)
            if pair:
                out.append(pair)
            if max_samples is not None and len(out) >= max_samples:
                break
        return out

    for obj in iter_jsonl(path):
        pair = _dpo_pair_from_obj(obj)
        if pair:
            out.append(pair)
        if max_samples is not None and len(out) >= max_samples:
            break
    return out

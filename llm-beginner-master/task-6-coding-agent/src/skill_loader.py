"""Skill loader：扫 SKILL.md，按 description 匹配，按需加载正文。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import yaml


def _parse_skill_md(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8")
    name = path.parent.name
    description = ""
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}
            name = str(meta.get("name") or name)
            description = str(meta.get("description") or "")
            body = parts[2].lstrip("\n")
    return {"name": name, "description": description, "body": body, "path": str(path)}


class SkillLoader:
    def __init__(self, skills_dir: str):
        self.skills_dir = Path(skills_dir)
        self._skills: Dict[str, Dict[str, str]] = {}
        if self.skills_dir.is_dir():
            for skill_md in sorted(self.skills_dir.glob("*/SKILL.md")):
                meta = _parse_skill_md(skill_md)
                # 索引时丢掉 body，按需再 load
                self._skills[meta["name"]] = {
                    "name": meta["name"],
                    "description": meta["description"],
                    "path": meta["path"],
                }

    def list_skills(self) -> List[dict]:
        return [
            {"name": s["name"], "description": s["description"]}
            for s in self._skills.values()
        ]

    def load(self, name: str) -> str:
        if name not in self._skills:
            raise KeyError(f"未知 skill: {name}")
        full = _parse_skill_md(Path(self._skills[name]["path"]))
        return full["body"]

    def match(self, query: str, top_k: int = 2) -> List[str]:
        q = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
        scored = []
        for s in self._skills.values():
            d = set(re.findall(r"[\w\u4e00-\u9fff]+", s["description"].lower()))
            score = len(q & d)
            # 名称命中再加点分
            if s["name"].replace("-", " ") in query.lower():
                score += 3
            scored.append((score, s["name"]))
        scored.sort(reverse=True)
        return [n for sc, n in scored if sc > 0][:top_k]

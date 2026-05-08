"""skills · 业务能力 (architecture.md §2.4)

层级: harness → skills → tools

一个 skill = 1 套指令包 + 一组 tool 引用 + 触发规则。
形态: skills_dir/<slug>/SKILL.md (frontmatter yaml + body markdown)。

SkillRegistry 扫 skills_dir · 解析 SKILL.md · 返 Skill 对象。
runtime tick 按 agent.skills 列表调 SkillRegistry.get(slug) · 拼到 system prompt + tools 列表。

env:
  - AKONG_SKILLS_DIR · 默认 ~/cast-skills/skills · `:` 分割支持多目录 (类 PATH · 后定义优先)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SKILLS_DIR = "~/cast-skills/skills"


class SkillError(Exception):
    """skill 加载 / 解析失败"""


@dataclass
class Skill:
    """一个 skill 的内存表示 (SKILL.md 解析结果)

    name        · slug · 跟目录名一致
    description · 一句话 · 含 TRIGGER / DO NOT TRIGGER (用于 LLM 自决是否跑)
    applies_to  · 哪类 agent 可装 (空 = 全适用)
    tools       · 该 skill 引的 tool id 列表 (例 cast.post · harness.update_memory)
    cooldown    · 同 agent 同 skill 触发间隔 (例 24h · never · 留接口 · runtime 暂不强制)
    prompt      · SKILL.md body (frontmatter 砍后) · 拼进 system prompt
    """

    name: str
    description: str
    applies_to: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    cooldown: str | None = None
    prompt: str = ""

    @property
    def slug(self) -> str:
        return self.name


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """从 SKILL.md 切出 yaml frontmatter + body

    格式:
        ---
        key: value
        ...
        ---
        # body markdown ...
    """
    if not text.startswith("---"):
        raise SkillError("SKILL.md must start with yaml frontmatter (---)")
    # 跳过开头的 ---\n
    rest = text[3:].lstrip("\n")
    end_idx = rest.find("\n---")
    if end_idx == -1:
        raise SkillError("SKILL.md frontmatter not closed (missing trailing ---)")
    fm_text = rest[:end_idx]
    body = rest[end_idx + 4 :].lstrip("\n")  # 跳过 \n---
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as e:
        raise SkillError(f"frontmatter yaml parse failed: {e}") from e
    if not isinstance(fm, dict):
        raise SkillError(f"frontmatter must be a mapping, got {type(fm).__name__}")
    return fm, body


def parse_skill_md(text: str, *, default_name: str | None = None) -> Skill:
    """解析单个 SKILL.md 文本 → Skill"""
    fm, body = _split_frontmatter(text)
    name = fm.get("name") or default_name
    if not name:
        raise SkillError("skill missing 'name' in frontmatter")
    description = fm.get("description") or ""
    applies_to = list(fm.get("applies_to") or [])
    tools = list(fm.get("tools") or [])
    cooldown = fm.get("cooldown")
    if cooldown is not None:
        cooldown = str(cooldown)
    return Skill(
        name=str(name),
        description=str(description),
        applies_to=[str(x) for x in applies_to],
        tools=[str(x) for x in tools],
        cooldown=cooldown,
        prompt=body,
    )


def _resolve_dirs(skills_dir: str | Path | list[str | Path] | None) -> list[Path]:
    """解析多目录配置 · `:` 分割 · ~ 展开 · 后定义优先 (覆盖前面同 slug)"""
    if skills_dir is None:
        raw = os.environ.get("AKONG_SKILLS_DIR", DEFAULT_SKILLS_DIR)
    elif isinstance(skills_dir, list):
        return [Path(p).expanduser() for p in skills_dir]
    else:
        raw = str(skills_dir)
    return [Path(p).expanduser() for p in raw.split(":") if p]


class SkillRegistry:
    """从一个或多个 skills_dir 加载 SKILL.md

    路径约定: <skills_dir>/<slug>/SKILL.md

    - load_all() 扫所有目录 · 返 {slug: Skill}
    - get(slug) 拿单个 · 不在返 None
    - 多目录时后定义覆盖前定义 (类 PATH 但反向 · 让用户 dir 覆盖默认)
    """

    def __init__(self, skills_dir: str | Path | list[str | Path] | None = None) -> None:
        self.dirs = _resolve_dirs(skills_dir)
        self._cache: dict[str, Skill] | None = None

    def load_all(self, *, force: bool = False) -> dict[str, Skill]:
        if self._cache is not None and not force:
            return self._cache
        result: dict[str, Skill] = {}
        for base in self.dirs:
            if not base.is_dir():
                continue
            for child in sorted(base.iterdir()):
                if not child.is_dir():
                    continue
                skill_md = child / "SKILL.md"
                if not skill_md.is_file():
                    continue
                try:
                    text = skill_md.read_text(encoding="utf-8")
                    skill = parse_skill_md(text, default_name=child.name)
                except SkillError:
                    # 损坏的 skill 不阻塞其它 · 但后续也许该 log
                    continue
                result[skill.slug] = skill
        self._cache = result
        return result

    def get(self, slug: str) -> Skill | None:
        return self.load_all().get(slug)

    def refresh(self) -> dict[str, Skill]:
        self._cache = None
        return self.load_all()

    def __contains__(self, slug: str) -> bool:
        return self.get(slug) is not None


# 进程级单例 (按 env 配 · runtime 默认用)
_default_registry: SkillRegistry | None = None


def default_registry() -> SkillRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = SkillRegistry()
    return _default_registry


def reset_default_registry() -> None:
    """测试 hook · 清掉单例"""
    global _default_registry
    _default_registry = None

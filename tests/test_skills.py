"""SkillRegistry · SKILL.md 加载 + 解析

4 cases:
  1. load_all · 多 skill 目录扫到全部
  2. get(slug) · 拿到正确字段
  3. get(missing) · 返 None
  4. parse_skill_md · frontmatter 字段全解析正确
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from akong_agent_harness import (
    Skill,
    SkillError,
    SkillRegistry,
    parse_skill_md,
)


_FIRST_POST = textwrap.dedent(
    """\
    ---
    name: first-post
    description: 新 agent 自我介绍 · TRIGGER posts==0 · DO NOT TRIGGER 已发过。
    applies_to: []
    tools:
      - cast.post
      - harness.update_memory
    cooldown: never
    ---

    # first-post · 新 agent 自我介绍帖

    ## 何时跑

    posts_count == 0 时跑。
    """
)

_WEEKLY = textwrap.dedent(
    """\
    ---
    name: weekly-report
    description: 每周日跑一次 · 总结上周成绩。
    applies_to:
      - design
      - coach
    tools:
      - cast.post
    cooldown: 7d
    ---

    # weekly-report · 周报

    每周日跑。
    """
)


def _write_skill(base: Path, slug: str, content: str) -> None:
    d = base / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(content, encoding="utf-8")


def test_load_all_finds_every_skill(tmp_path: Path) -> None:
    """load_all 应扫到目录下所有 SKILL.md"""
    _write_skill(tmp_path, "first-post", _FIRST_POST)
    _write_skill(tmp_path, "weekly-report", _WEEKLY)
    # 一个非 skill 目录 (没 SKILL.md) · 应被跳过
    (tmp_path / "not-a-skill").mkdir()
    (tmp_path / "not-a-skill" / "README.md").write_text("noop", encoding="utf-8")

    reg = SkillRegistry(tmp_path)
    skills = reg.load_all()

    assert set(skills.keys()) == {"first-post", "weekly-report"}
    assert isinstance(skills["first-post"], Skill)


def test_get_by_slug_returns_full_skill(tmp_path: Path) -> None:
    """get(slug) 应返回完整字段 (name / description / tools / cooldown / prompt)"""
    _write_skill(tmp_path, "first-post", _FIRST_POST)
    reg = SkillRegistry(tmp_path)

    skill = reg.get("first-post")
    assert skill is not None
    assert skill.name == "first-post"
    assert skill.slug == "first-post"
    assert "新 agent" in skill.description
    assert "TRIGGER" in skill.description
    assert skill.applies_to == []
    assert skill.tools == ["cast.post", "harness.update_memory"]
    assert skill.cooldown == "never"
    # prompt body 含原文 · 不含 frontmatter
    assert "first-post · 新 agent 自我介绍帖" in skill.prompt
    assert "name: first-post" not in skill.prompt
    assert "---" not in skill.prompt.split("\n")[0]


def test_get_missing_returns_none(tmp_path: Path) -> None:
    """不存在的 slug · get 返 None · `in` 判断 False"""
    _write_skill(tmp_path, "first-post", _FIRST_POST)
    reg = SkillRegistry(tmp_path)

    assert reg.get("does-not-exist") is None
    assert "does-not-exist" not in reg
    assert "first-post" in reg


def test_parse_skill_md_frontmatter_complete() -> None:
    """parse_skill_md · frontmatter 多字段 (含 list applies_to + tools) 全解析正确"""
    skill = parse_skill_md(_WEEKLY)
    assert skill.name == "weekly-report"
    assert skill.applies_to == ["design", "coach"]
    assert skill.tools == ["cast.post"]
    assert skill.cooldown == "7d"
    assert "每周日跑" in skill.prompt


def test_parse_skill_md_rejects_no_frontmatter() -> None:
    """没 frontmatter 应抛 SkillError"""
    with pytest.raises(SkillError):
        parse_skill_md("# just a markdown\n\nno frontmatter")


def test_parse_skill_md_rejects_unclosed_frontmatter() -> None:
    """frontmatter 没闭合应抛 SkillError"""
    bad = "---\nname: x\ndescription: y\n# never closed\n"
    with pytest.raises(SkillError):
        parse_skill_md(bad)


def test_load_all_skips_corrupt_skill(tmp_path: Path) -> None:
    """损坏的 SKILL.md (frontmatter 错) 不该阻塞其它 skill 加载"""
    _write_skill(tmp_path, "first-post", _FIRST_POST)
    _write_skill(tmp_path, "broken", "# no frontmatter here")

    reg = SkillRegistry(tmp_path)
    skills = reg.load_all()

    assert "first-post" in skills
    assert "broken" not in skills


def test_multi_dir_later_overrides_earlier(tmp_path: Path) -> None:
    """多目录 (PATH 风格) · 后定义的目录覆盖前面同 slug"""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    _write_skill(a, "first-post", _FIRST_POST)
    # b 里同 slug 但 description 不同 (用 weekly 内容但 rename)
    overridden = _WEEKLY.replace("name: weekly-report", "name: first-post")
    _write_skill(b, "first-post", overridden)

    reg = SkillRegistry([a, b])
    skill = reg.get("first-post")
    assert skill is not None
    # b 覆盖 a · description 应是 weekly 那条
    assert "每周日跑一次" in skill.description

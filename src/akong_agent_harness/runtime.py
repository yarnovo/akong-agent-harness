"""runtime · harness 主循环 (architecture.md §2.7)

`tick(agent_id, trigger)` 跑一轮:
  1. 通过 cast-api 拉 6 件套 (identity / playbook / memory recent / tools list / state)
  2. 拼 system prompt (soul + playbook + style + recent memory + 当前 trigger)
  3. 调 LLM (function calling · tools 来自 Tools.list()) · OpenAI SDK 兼容 endpoint
  4. 执行 tool calls (Tools.call · 路由到平台仓注册实现)
  5. 写 memory (LLM 调 harness.update_memory · harness 自带)
  6. 写 change_log (LLM 调 harness.update_self · harness 自带)
  7. 决定 next wakeup
  8. 返 TickResult

DeepSeek-v3.1 注: thinking mode 跟 function calling 互斥 · 必 enable_thinking=False。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
from openai import OpenAI

from .memory import DEFAULT_API_BASE_URL, Memory, RdsAdapter, MemoryEntry
from .skills import Skill, SkillRegistry, default_registry as _default_skill_registry
from .tools import ToolNotRegisteredError, ToolSpec, Tools


DEFAULT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_MODEL = "deepseek-v3.1"

# tool_id → OpenAI function name 编码: . → __ (OpenAI 不允许 . 在 function name)
_TOOL_NAME_SEP = "__"


def _encode_tool_name(tool_id: str) -> str:
    return tool_id.replace(".", _TOOL_NAME_SEP)


def _decode_tool_name(fn_name: str) -> str:
    return fn_name.replace(_TOOL_NAME_SEP, ".")


@dataclass
class Trigger:
    kind: str  # 'cron' | 'event' | 'human-dm' | 'manual'
    payload: dict[str, Any] | None = None


@dataclass
class TickResult:
    actions: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    next_wakeup: datetime | None = None
    stopped: bool = False
    error: str | None = None


@dataclass
class _AgentBundle:
    """从 cast-api 拉出来的 6 件套 (identity + playbook + memory + tools + state) + skills"""

    agent: dict[str, Any]
    memory_recent: list[MemoryEntry]
    tools: list[ToolSpec]
    skills: list[Skill] = field(default_factory=list)


def _resolve_agent_skill_slugs(agent: dict[str, Any]) -> list[str]:
    """从 agent record 解析 skills slug 列表

    cast-api 当前没原生 skills 字段 (走 metadata_json 字段) · 也兼容 agent['skills'] 直传。
    """
    raw = agent.get("skills")
    if raw is None:
        meta = agent.get("metadata_json")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        if isinstance(meta, dict):
            raw = meta.get("skills")
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(s) for s in raw]
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    return []


def _load_skills_for_agent(
    agent: dict[str, Any],
    registry: SkillRegistry,
) -> list[Skill]:
    """按 agent.skills slug 列表 · 调 registry.get · 拿到 Skill 对象 (不存在静默跳过)"""
    skills: list[Skill] = []
    for slug in _resolve_agent_skill_slugs(agent):
        skill = registry.get(slug)
        if skill is not None:
            skills.append(skill)
    return skills


def _fetch_agent_bundle(
    agent_id: str,
    api_base_url: str,
    memory: Memory,
    tools: Tools,
    *,
    memory_recent_limit: int = 20,
    timeout: float = 10.0,
    skill_registry: SkillRegistry | None = None,
) -> _AgentBundle:
    with httpx.Client(base_url=api_base_url.rstrip("/"), timeout=timeout, trust_env=False) as client:
        resp = client.get(f"/api/agents/{agent_id}")
        if resp.status_code >= 400:
            raise RuntimeError(f"GET /api/agents/{agent_id} → {resp.status_code} {resp.text[:200]}")
        agent_data = resp.json()
    recent = memory.recent(limit=memory_recent_limit)
    tool_specs = tools.list()
    registry = skill_registry or _default_skill_registry()
    skills = _load_skills_for_agent(agent_data, registry)
    return _AgentBundle(agent=agent_data, memory_recent=recent, tools=tool_specs, skills=skills)


def _build_system_prompt(bundle: _AgentBundle) -> str:
    a = bundle.agent
    parts: list[str] = []
    parts.append(f"# 你是 {a.get('name', 'agent')} (id={a.get('id')})")
    if a.get("tagline"):
        parts.append(f"\n**简介**: {a['tagline']}")
    if a.get("soul"):
        parts.append(f"\n## soul · 人设\n\n{a['soul']}")
    if a.get("playbook"):
        parts.append(f"\n## playbook · 守则\n\n{a['playbook']}")
    if a.get("style"):
        parts.append(f"\n## style · 文风\n\n{a['style']}")
    if bundle.memory_recent:
        parts.append("\n## 最近记忆 (倒序)\n")
        for m in bundle.memory_recent:
            ts = m.created_at.isoformat() if isinstance(m.created_at, datetime) else m.created_at
            parts.append(f"- [{m.kind} · {ts}] {m.content}")
    if bundle.skills:
        parts.append("\n## 已装载 skills (业务能力 · 按需触发)\n")
        for skill in bundle.skills:
            parts.append(f"\n## skill: {skill.name}\n")
            if skill.description:
                parts.append(f"_{skill.description}_\n")
            if skill.prompt:
                parts.append(skill.prompt)
    return "\n".join(parts)


def _merge_tool_specs(
    platform_specs: list[ToolSpec],
    skills: list[Skill],
) -> list[ToolSpec]:
    """合并 6 件套基础 tools + skill 引的 tools (去重 · platform spec 优先)

    skill 只声明 tool id 字符串 · 真 spec 来自平台 registry (cast-api 返回)。
    若 skill 引了 platform 没注册的 tool id · 静默忽略 (LLM tool 列表里不出现)。
    """
    by_id: dict[str, ToolSpec] = {s.id: s for s in platform_specs}
    available = {s.id for s in platform_specs}
    referenced: set[str] = set()
    for skill in skills:
        for tool_id in skill.tools:
            if tool_id.startswith("harness."):
                # harness 自带 tool · runtime 单独注 builtin · 不进 spec list
                continue
            referenced.add(tool_id)
    # 当前实现: skill 引的 tool 必须在 platform spec 里 · 没找到的丢
    final: list[ToolSpec] = []
    seen: set[str] = set()
    for spec in platform_specs:
        if spec.id in seen:
            continue
        seen.add(spec.id)
        final.append(spec)
    # skill 引的 tool 默认就在 platform_specs 里 (兼集) · 不再重复加
    # 留扩展位: 未来 skill 可独立带 spec 时这里 union
    _ = referenced, available, by_id
    return final


def _build_user_prompt(trigger: Trigger) -> str:
    payload_str = json.dumps(trigger.payload or {}, ensure_ascii=False, default=str)
    return f"trigger: {trigger.kind}\npayload: {payload_str}\n\n请决定如何回应 · 必要时调 tool · 然后调 set_next_wakeup 或 stop_for_now 结束。"


# === 内置 harness tools (6 件套之 tools 一部分 · 由 runtime 直接处理) ===


def _builtin_tools() -> list[dict[str, Any]]:
    """这 4 个 tool harness 自带 · 不走平台 registry · runtime 直接处理"""
    return [
        {
            "type": "function",
            "function": {
                "name": "harness__update_memory",
                "description": "把一条新长记忆写进 agent 的记忆 log (event/learning/relationship/preference 等)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "description": "event | learning | relationship | preference"},
                        "content": {"type": "string", "description": "markdown · 自由格式"},
                    },
                    "required": ["kind", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "harness__update_self",
                "description": "改自己的 soul / playbook / style / rules_json / metadata_json (自演化 · 写 change_log)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                            "enum": ["soul", "playbook", "style", "rules_json", "metadata_json"],
                        },
                        "new_value": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["field", "new_value"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "harness__set_next_wakeup",
                "description": "告诉 runtime 下次什么时候叫我醒 (ISO 8601 datetime)",
                "parameters": {
                    "type": "object",
                    "properties": {"at": {"type": "string", "description": "ISO 8601 datetime · UTC"}},
                    "required": ["at"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "harness__stop_for_now",
                "description": "本次 tick 完成 · 不需要 next wakeup (event 触发的 agent 等下一个 trigger)",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def _handle_builtin(
    fn_name: str,
    args: dict[str, Any],
    *,
    agent_id: str,
    memory: Memory,
    api_base_url: str,
    state: dict[str, Any],
) -> tuple[bool, Any]:
    """返 (handled, result) · handled=True 表示是 builtin · runtime 不再走平台 registry"""
    if fn_name == "harness__update_memory":
        entry = memory.append(kind=args.get("kind", "event"), content=args.get("content", ""))
        return True, {"ok": True, "memory_id": entry.id}
    if fn_name == "harness__update_self":
        body = {
            "field": args["field"],
            "new_value": args.get("new_value"),
            "reason": args.get("reason"),
            "changed_by": "self",
        }
        with httpx.Client(base_url=api_base_url.rstrip("/"), timeout=10.0, trust_env=False) as client:
            resp = client.post(f"/api/agents/{agent_id}/update-self", json=body)
        if resp.status_code >= 400:
            return True, {"ok": False, "error": f"{resp.status_code} {resp.text[:200]}"}
        return True, {"ok": True}
    if fn_name == "harness__set_next_wakeup":
        at = args.get("at", "")
        try:
            from dateutil.parser import isoparse

            state["next_wakeup"] = isoparse(at)
            state["stopped"] = False
        except Exception as e:
            return True, {"ok": False, "error": f"bad datetime: {e}"}
        return True, {"ok": True, "next_wakeup": at}
    if fn_name == "harness__stop_for_now":
        state["stopped"] = True
        state["next_wakeup"] = None
        return True, {"ok": True}
    return False, None


def tick(
    agent_id: str,
    trigger: Trigger,
    *,
    max_turns: int = 5,
    api_base_url: str | None = None,
    llm_api_key: str | None = None,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    memory: Memory | None = None,
    tools: Tools | None = None,
    openai_client: Any | None = None,
    skill_registry: SkillRegistry | None = None,
) -> TickResult:
    """harness 主循环 · 跑一轮 agent。

    生产用法只传 agent_id + trigger · 其他参数走 env / 默认。
    测试可注入 memory / tools / openai_client / skill_registry。
    """
    api_base_url = (api_base_url or os.environ.get("AKONG_API_BASE_URL", DEFAULT_API_BASE_URL)).rstrip("/")
    llm_api_key = llm_api_key or os.environ.get("AKONG_LLM_API_KEY", "")
    llm_base_url = llm_base_url or os.environ.get("AKONG_LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    llm_model = llm_model or os.environ.get("AKONG_LLM_MODEL", DEFAULT_LLM_MODEL)

    if memory is None:
        memory = RdsAdapter(agent_id, api_base_url=api_base_url)
    if tools is None:
        tools = Tools.connect(agent_id, api_base_url=api_base_url)
    if openai_client is None:
        if not llm_api_key:
            raise RuntimeError("AKONG_LLM_API_KEY required (or pass openai_client=)")
        openai_client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)

    result = TickResult()

    try:
        bundle = _fetch_agent_bundle(
            agent_id,
            api_base_url,
            memory,
            tools,
            skill_registry=skill_registry,
        )
    except Exception as e:
        result.error = f"fetch bundle failed: {e}"
        return result

    system_prompt = _build_system_prompt(bundle)
    user_prompt = _build_user_prompt(trigger)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result.messages = list(messages)

    # 拼 tools list = 平台 tools + skill 引的 tools (去重 · merged) + 4 个 harness builtin
    merged_specs = _merge_tool_specs(bundle.tools, bundle.skills)
    tool_specs_openai = [s.to_openai_function() for s in merged_specs] + _builtin_tools()

    state = {"next_wakeup": None, "stopped": False}

    # DeepSeek-v3.1 thinking 跟 function calling 互斥 · 必 enable_thinking=False
    extra_body = {"enable_thinking": False} if "deepseek" in llm_model.lower() else {}

    for turn in range(max_turns):
        try:
            completion = openai_client.chat.completions.create(
                model=llm_model,
                messages=messages,
                tools=tool_specs_openai if tool_specs_openai else None,
                tool_choice="auto" if tool_specs_openai else None,
                extra_body=extra_body or None,
            )
        except Exception as e:
            result.error = f"LLM call failed (turn {turn}): {e}"
            return result

        choice = completion.choices[0]
        msg = choice.message
        msg_dict: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(msg_dict)
        result.messages.append(msg_dict)

        if not tool_calls:
            # LLM 没调 tool · 视为本轮结束 · 默认 stop
            state["stopped"] = True
            break

        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            handled, res = _handle_builtin(
                fn_name,
                args,
                agent_id=agent_id,
                memory=memory,
                api_base_url=api_base_url,
                state=state,
            )
            if not handled:
                tool_id = _decode_tool_name(fn_name)
                try:
                    res = tools.call(tool_id, **args)
                except ToolNotRegisteredError as e:
                    res = {"ok": False, "error": str(e)}
                except Exception as e:
                    res = {"ok": False, "error": f"{type(e).__name__}: {e}"}

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(res, ensure_ascii=False, default=str),
            }
            messages.append(tool_msg)
            result.messages.append(tool_msg)
            result.actions.append(
                {"tool_id": _decode_tool_name(fn_name), "args": args, "result": res}
            )

        if state["stopped"]:
            break

    result.next_wakeup = state["next_wakeup"]
    result.stopped = bool(state["stopped"])
    return result

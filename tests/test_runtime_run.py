"""runtime.run() 多轮 tool use loop 集成测 · mock LLMClient + InMemorySession + tool error 自纠"""

from __future__ import annotations

import json
from typing import Any

import pytest

from akong_agent_harness import (
    AgentDef,
    InMemorySession,
    RunResult,
    Tools,
    register_tool,
    run,
)
from akong_agent_harness.llm import ChatResponse, ToolCall, Usage
from akong_agent_harness.tools import clear_registered_tools, ToolSpec


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registered_tools()
    yield
    clear_registered_tools()


class _ScriptedLLM:
    """注入式 LLMClient · 按顺序返预设 ChatResponse"""

    model_id = "scripted"

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat_completion(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._responses:
            return ChatResponse(content="(no more)", stop_reason="end_turn")
        return self._responses.pop(0)


class _StaticTools(Tools):
    """跳过 cast-api · 直接返预设 tool spec"""

    def __init__(self, specs: list[ToolSpec]) -> None:
        self._specs = specs
        self._cache = specs
        self.agent_id = "ag_test"
        self.platform = None
        self.api_base_url = "http://fake"

    def list(self):
        return self._specs


def _spec(tool_id: str) -> ToolSpec:
    return ToolSpec(
        id=tool_id,
        name=tool_id.split(".")[-1],
        description=f"test {tool_id}",
        params_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        returns_schema={},
        platform="test",
        scope="normal",
    )


# === 基本: 1 轮 LLM 直接 end_turn ===


async def test_run_single_turn_no_tool():
    agent = AgentDef(id="ag_test", name="Test")
    session = InMemorySession("sess_1")
    llm = _ScriptedLLM(
        [ChatResponse(content="hello back", stop_reason="end_turn", usage=Usage(10, 5, 15))]
    )
    result: RunResult = await run(
        agent=agent,
        user_message="hi",
        session=session,
        llm_client=llm,
        max_turns=10,
    )
    assert result.error is None
    assert result.stop_reason == "end_turn"
    assert result.final_text == "hello back"
    assert result.turns_used == 1
    assert result.usage["total_tokens"] == 15

    # session 持久化: user + assistant
    msgs = await session.load()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


# === 多轮 tool use loop ===


async def test_run_tool_use_loop_then_end():
    @register_tool("test.echo")
    def echo(x: str) -> dict:
        return {"echoed": x}

    agent = AgentDef(id="ag_test", name="Test")
    session = InMemorySession("sess_2")
    tools = _StaticTools([_spec("test.echo")])

    llm = _ScriptedLLM(
        [
            # turn 1: LLM 调 test.echo
            ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(id="call_1", name="test__echo", arguments=json.dumps({"x": "ping"}))
                ],
                stop_reason="tool_use",
                usage=Usage(10, 5, 15),
            ),
            # turn 2: LLM 看见 result · 自然结束
            ChatResponse(
                content="echo done", stop_reason="end_turn", usage=Usage(20, 7, 27)
            ),
        ]
    )

    result = await run(
        agent=agent,
        user_message="echo ping",
        session=session,
        llm_client=llm,
        tools=tools,
        max_turns=5,
    )
    assert result.error is None
    assert result.stop_reason == "end_turn"
    assert result.final_text == "echo done"
    assert result.turns_used == 2
    assert len(result.actions) == 1
    assert result.actions[0]["tool_id"] == "test.echo"
    assert result.actions[0]["result"] == {"echoed": "ping"}

    # session: user + assistant1(tool_call) + tool_result + assistant2
    msgs = await session.load()
    assert len(msgs) == 4
    assert msgs[1]["tool_calls"][0]["id"] == "call_1"
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == "call_1"
    # token 累加
    assert result.usage["total_tokens"] == 42


# === max_turns 截止 ===


async def test_run_max_turns_caps():
    """LLM 一直返 tool_use · max_turns 到上限 stop"""

    @register_tool("test.loop")
    def loop_tool() -> dict:
        return {"keep_going": True}

    agent = AgentDef(id="ag_test")
    session = InMemorySession("sess_3")
    tools = _StaticTools([_spec("test.loop")])

    # 让 LLM 永远返 tool_use · 看 run() 是否在 max_turns=2 截止
    def make_tc():
        return ChatResponse(
            tool_calls=[ToolCall(id=f"call_{id(object())}", name="test__loop", arguments="{}")],
            stop_reason="tool_use",
        )

    llm = _ScriptedLLM([make_tc(), make_tc(), make_tc(), make_tc()])

    result = await run(
        agent=agent,
        user_message="loop",
        session=session,
        llm_client=llm,
        tools=tools,
        max_turns=2,
    )
    assert result.stop_reason == "max_turns"
    assert result.turns_used == 2


# === harness builtin: stop_for_now → harness_stop ===


async def test_run_harness_stop_for_now():
    agent = AgentDef(id="ag_test")
    session = InMemorySession("sess_4")
    llm = _ScriptedLLM(
        [
            ChatResponse(
                tool_calls=[
                    ToolCall(id="call_s", name="harness__stop_for_now", arguments="{}"),
                ],
                stop_reason="tool_use",
            )
        ]
    )
    result = await run(
        agent=agent,
        user_message="stop please",
        session=session,
        llm_client=llm,
        max_turns=5,
    )
    assert result.stop_reason == "harness_stop"
    # builtin tool 也进 actions
    assert any(a["tool_id"] == "harness.stop_for_now" for a in result.actions)


# === tool 执行错误 → result 写回 session 让 LLM 自纠 ===


async def test_run_tool_error_self_correct():
    """tool 抛异常 · runtime 把 error 当 tool_result append · LLM 看到能自纠"""

    @register_tool("test.broken")
    def broken() -> dict:
        raise ValueError("backend boom")

    agent = AgentDef(id="ag_test")
    session = InMemorySession("sess_5")
    tools = _StaticTools([_spec("test.broken")])

    llm = _ScriptedLLM(
        [
            # turn 1: 调 broken · 会 raise
            ChatResponse(
                tool_calls=[ToolCall(id="call_b", name="test__broken", arguments="{}")],
                stop_reason="tool_use",
            ),
            # turn 2: LLM 看到 error · 道歉收手
            ChatResponse(content="sorry, tool failed", stop_reason="end_turn"),
        ]
    )

    result = await run(
        agent=agent,
        user_message="run broken",
        session=session,
        llm_client=llm,
        tools=tools,
        max_turns=5,
    )
    assert result.error is None  # run 整体没崩
    assert result.stop_reason == "end_turn"
    assert result.final_text == "sorry, tool failed"

    # tool result 含 error · LLM 第二轮看到的 messages 必含
    second_call_msgs = llm.calls[1]["messages"]
    tool_msg = [m for m in second_call_msgs if m["role"] == "tool"][0]
    payload = json.loads(tool_msg["content"])
    assert payload["ok"] is False
    assert "backend boom" in payload["error"]


# === ToolNotRegistered → 同样自纠 ===


async def test_run_tool_not_registered():
    agent = AgentDef(id="ag_test")
    session = InMemorySession("sess_6")
    tools = _StaticTools([_spec("test.ghost")])  # spec 有 · 实现没 register

    llm = _ScriptedLLM(
        [
            ChatResponse(
                tool_calls=[ToolCall(id="call_g", name="test__ghost", arguments="{}")],
                stop_reason="tool_use",
            ),
            ChatResponse(content="abandoning ghost tool", stop_reason="end_turn"),
        ]
    )

    result = await run(
        agent=agent,
        user_message="try ghost",
        session=session,
        llm_client=llm,
        tools=tools,
        max_turns=3,
    )
    assert result.error is None
    assert result.stop_reason == "end_turn"
    # tool result 标 ok=False
    action = result.actions[0]
    assert action["result"]["ok"] is False
    assert "no registered implementation" in action["result"]["error"]


# === session history 真续上 (跨 run 调用) ===


async def test_run_loads_existing_session_history():
    agent = AgentDef(id="ag_test")
    session = InMemorySession(
        "sess_7",
        initial=[
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ],
    )
    llm = _ScriptedLLM([ChatResponse(content="续上下文", stop_reason="end_turn")])

    result = await run(
        agent=agent,
        user_message="next",
        session=session,
        llm_client=llm,
        max_turns=2,
    )
    assert result.error is None
    # LLM 第 1 轮看到的 messages 必须含老的 2 条 + system + 新 user
    msgs_seen = llm.calls[0]["messages"]
    contents = [m.get("content") for m in msgs_seen]
    assert "old question" in contents
    assert "old answer" in contents
    assert "next" in contents

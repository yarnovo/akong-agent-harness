"""runtime.tick · 端到端 smoke (mock LLM + mock cast-api + mock tool 注册)"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from akong_agent_harness import RdsAdapter, Tools, Trigger, register_tool, tick
from akong_agent_harness.tools import clear_registered_tools


@dataclass
class _ToolCall:
    id: str
    function_name: str
    function_arguments: str


class _FakeMessage:
    def __init__(self, content: str = "", tool_calls: list[_ToolCall] | None = None) -> None:
        self.content = content
        if tool_calls is None:
            self.tool_calls = None
        else:
            self.tool_calls = [_FakeToolCall(tc) for tc in tool_calls]


class _FakeToolCall:
    def __init__(self, raw: _ToolCall) -> None:
        self.id = raw.id
        self.type = "function"
        self.function = _FakeFn(raw.function_name, raw.function_arguments)


class _FakeFn:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeChoice:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeCompletion:
    def __init__(self, choices: list[_FakeChoice]) -> None:
        self.choices = choices


class _FakeChat:
    def __init__(self, scripted: list[_FakeMessage]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    @property
    def completions(self) -> "_FakeChat":
        return self

    def create(self, **kwargs: Any) -> _FakeCompletion:
        self.calls.append(kwargs)
        if not self._scripted:
            # 默认结尾 · 没 tool call
            return _FakeCompletion([_FakeChoice(_FakeMessage(content="done"))])
        msg = self._scripted.pop(0)
        return _FakeCompletion([_FakeChoice(msg)])


class _FakeOpenAI:
    def __init__(self, scripted: list[_FakeMessage]) -> None:
        self.chat = _FakeChat(scripted)


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registered_tools()
    yield
    clear_registered_tools()


def _agent_row():
    return {
        "id": "ag_alice",
        "name": "Alice",
        "tagline": "热心咨询师",
        "soul": "温和 + 专业",
        "playbook": "24h 内回",
        "style": "口语",
        "expertise": "情感",
        "owner_id": "u_owner",
        "persona": {"id": "u_alice", "name": "Alice", "avatar": "", "bio": "热心咨询师", "location": None},
        "starting_price_cents": None,
        "services_count": 0,
        "status": "active",
        "services": [],
        "created_at": "2026-05-01T00:00:00+00:00",
        "role": "normal",
        "rules_json": None,
        "metadata_json": None,
    }


def _build_cast_api_transport(memories_capture: list, change_capture: list) -> httpx.MockTransport:
    """模拟 cast-api · 返 agent + 空 memories + 1 个 tool · 接受 update-self"""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/agents/ag_alice":
            return httpx.Response(200, json=_agent_row(), request=request)
        if path == "/api/agents/ag_alice/memories":
            if request.method == "POST":
                body = json.loads(request.content.decode() or "{}")
                row = {
                    "id": f"mem_{len(memories_capture) + 1}",
                    "agent_id": "ag_alice",
                    "kind": body["kind"],
                    "content": body["content"],
                    "created_at": "2026-05-08T10:00:00+00:00",
                }
                memories_capture.append(row)
                return httpx.Response(201, json=row, request=request)
            # GET
            return httpx.Response(200, json=memories_capture, request=request)
        if path == "/api/agents/ag_alice/tools":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "cast.send_dm",
                        "name": "send_dm",
                        "description": "私信",
                        "params_schema_json": json.dumps(
                            {
                                "type": "object",
                                "properties": {
                                    "to_user_id": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["to_user_id", "content"],
                            }
                        ),
                        "returns_schema_json": "{}",
                        "platform": "cast",
                        "scope": "normal",
                    }
                ],
                request=request,
            )
        if path == "/api/agents/ag_alice/update-self":
            change_capture.append(json.loads(request.content.decode()))
            return httpx.Response(200, json=_agent_row(), request=request)
        return httpx.Response(404, json={"detail": f"unmocked {path}"}, request=request)

    return httpx.MockTransport(handler)


def test_tick_smoke_runs_tool_then_stops():
    @register_tool("cast.send_dm")
    def send_dm(to_user_id: str, content: str) -> dict:
        return {"sent": True, "to": to_user_id}

    memories: list = []
    changes: list = []
    transport = _build_cast_api_transport(memories, changes)

    # 注入 memory + tools 用同 transport
    mem = RdsAdapter("ag_alice", api_base_url="http://fake")
    mem._client = httpx.Client(base_url="http://fake", transport=transport)
    tools = Tools.connect("ag_alice", platform="cast", api_base_url="http://fake")
    tools._client = httpx.Client(base_url="http://fake", transport=transport)

    # 把 _fetch_agent_bundle 内部用 httpx.Client 也走 mock transport
    import akong_agent_harness.runtime as runtime_mod

    orig_client = httpx.Client

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_client(*args, **kwargs)

    runtime_mod.httpx.Client = patched_client  # type: ignore[assignment]

    try:
        scripted = [
            # turn 1 · LLM 调 cast.send_dm + harness.update_memory
            _FakeMessage(
                tool_calls=[
                    _ToolCall(
                        id="call_1",
                        function_name="cast__send_dm",
                        function_arguments=json.dumps({"to_user_id": "u_buyer", "content": "hi"}),
                    ),
                    _ToolCall(
                        id="call_2",
                        function_name="harness__update_memory",
                        function_arguments=json.dumps({"kind": "event", "content": "回了 buyer"}),
                    ),
                ]
            ),
            # turn 2 · LLM 调 stop_for_now
            _FakeMessage(
                tool_calls=[
                    _ToolCall(
                        id="call_3",
                        function_name="harness__stop_for_now",
                        function_arguments="{}",
                    )
                ]
            ),
        ]
        fake_llm = _FakeOpenAI(scripted)
        result = tick(
            "ag_alice",
            Trigger(kind="manual", payload={"text": "ping"}),
            api_base_url="http://fake",
            memory=mem,
            tools=tools,
            openai_client=fake_llm,
            llm_model="deepseek-v3.1",
        )
    finally:
        runtime_mod.httpx.Client = orig_client  # type: ignore[assignment]

    assert result.error is None, f"unexpected error: {result.error}"
    assert result.stopped is True
    # 至少 3 个 action: send_dm + update_memory + stop_for_now
    tool_ids = [a["tool_id"] for a in result.actions]
    assert "cast.send_dm" in tool_ids
    assert "harness.update_memory" in tool_ids
    assert "harness.stop_for_now" in tool_ids
    # memory 真写进去
    assert len(memories) == 1
    assert memories[0]["content"] == "回了 buyer"
    # LLM extra_body 带 enable_thinking=False (DeepSeek)
    assert fake_llm.chat.calls[0].get("extra_body") == {"enable_thinking": False}

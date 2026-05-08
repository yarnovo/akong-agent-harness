"""Session 单测 · InMemorySession + RdsSession (httpx MockTransport)"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from akong_agent_harness.session import (
    InMemorySession,
    RdsSession,
    SessionError,
    SessionUnavailable,
)


# === InMemorySession ===


async def test_in_memory_session_lifecycle():
    s = InMemorySession("sess_1")
    assert s.session_id == "sess_1"
    assert await s.load() == []

    await s.append({"role": "user", "content": "hi"})
    await s.append({"role": "assistant", "content": "hello"})
    msgs = await s.load()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["content"] == "hello"

    # load 是 copy · 改返回值不影响内部
    msgs[0]["content"] = "modified"
    msgs2 = await s.load()
    assert msgs2[0]["content"] == "hi"

    await s.clear()
    assert await s.load() == []


async def test_in_memory_session_validates_message():
    s = InMemorySession("sess_x")
    with pytest.raises(TypeError):
        await s.append("not a dict")  # type: ignore
    with pytest.raises(ValueError):
        await s.append({"content": "no role"})


async def test_in_memory_session_with_tool_calls():
    s = InMemorySession("sess_2")
    await s.append({"role": "user", "content": "go"})
    await s.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        }
    )
    await s.append({"role": "tool", "tool_call_id": "tc_1", "content": '{"ok": true}'})
    msgs = await s.load()
    assert len(msgs) == 3
    assert msgs[1]["tool_calls"][0]["id"] == "tc_1"
    assert msgs[2]["tool_call_id"] == "tc_1"


# === RdsSession (cast-api · httpx mock) ===


def _build_chat_messages_transport(store: list[dict[str, Any]], *, exists: bool = True) -> httpx.MockTransport:
    """模拟 cast-api /api/chat_messages endpoint。

    exists=False → 全部 404 (模拟 endpoint 还没上线)
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path != "/api/chat_messages":
            return httpx.Response(404, json={"detail": "not found"}, request=request)
        if not exists:
            return httpx.Response(404, json={"detail": "endpoint not deployed"}, request=request)
        if request.method == "POST":
            body = json.loads(request.content.decode() or "{}")
            row = {
                "id": f"msg_{len(store) + 1}",
                "session_id": body["session_id"],
                "role": body["role"],
                "content": body.get("content") or "",
                "tool_calls": body.get("tool_calls"),
                "tool_call_id": body.get("tool_call_id"),
                "agent_id": body.get("agent_id"),
                "created_at": "2026-05-08T10:00:00+00:00",
            }
            store.append(row)
            return httpx.Response(201, json=row, request=request)
        if request.method == "GET":
            sid = request.url.params.get("session_id")
            rows = [r for r in store if r["session_id"] == sid]
            return httpx.Response(200, json=rows, request=request)
        if request.method == "DELETE":
            sid = request.url.params.get("session_id")
            store[:] = [r for r in store if r["session_id"] != sid]
            return httpx.Response(204, request=request)
        return httpx.Response(405, request=request)

    return httpx.MockTransport(handler)


async def test_rds_session_append_and_load():
    store: list[dict[str, Any]] = []
    transport = _build_chat_messages_transport(store)
    client = httpx.AsyncClient(base_url="http://fake", transport=transport)

    s = RdsSession("sess_rds_1", api_base_url="http://fake", agent_id="ag_x", client=client)
    assert await s.load() == []

    await s.append({"role": "user", "content": "hi"})
    await s.append({"role": "assistant", "content": "hello"})
    msgs = await s.load()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hi"
    assert msgs[1]["content"] == "hello"
    # store 真持久化 (跨 RdsSession 实例 · 模拟跨 FC invoke)
    assert store[0]["agent_id"] == "ag_x"

    await client.aclose()


async def test_rds_session_with_tool_calls_roundtrip():
    store: list[dict[str, Any]] = []
    transport = _build_chat_messages_transport(store)
    client = httpx.AsyncClient(base_url="http://fake", transport=transport)
    s = RdsSession("sess_rds_2", api_base_url="http://fake", client=client)

    await s.append({"role": "user", "content": "go"})
    await s.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "tc_a", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        }
    )
    await s.append({"role": "tool", "tool_call_id": "tc_a", "content": '{"ok": true}'})

    msgs = await s.load()
    assert len(msgs) == 3
    assert msgs[1]["tool_calls"][0]["id"] == "tc_a"
    assert msgs[2]["tool_call_id"] == "tc_a"
    await client.aclose()


async def test_rds_session_unavailable_on_404():
    transport = _build_chat_messages_transport([], exists=False)
    client = httpx.AsyncClient(base_url="http://fake", transport=transport)
    s = RdsSession("sess_rds_x", api_base_url="http://fake", client=client)

    with pytest.raises(SessionUnavailable):
        await s.append({"role": "user", "content": "hi"})

    with pytest.raises(SessionUnavailable):
        await s.load()
    await client.aclose()


async def test_rds_session_clear():
    store: list[dict[str, Any]] = []
    transport = _build_chat_messages_transport(store)
    client = httpx.AsyncClient(base_url="http://fake", transport=transport)
    s = RdsSession("sess_clear", api_base_url="http://fake", client=client)

    await s.append({"role": "user", "content": "hi"})
    assert len(await s.load()) == 1
    await s.clear()
    assert await s.load() == []
    await client.aclose()

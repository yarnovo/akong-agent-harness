"""Memory RdsAdapter · mock cast-api HTTP · 测 append + recent + search"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest

from akong_agent_harness import RdsAdapter


def _api_response(status_code: int, json_body):
    req = httpx.Request("GET", "http://test")
    return httpx.Response(status_code, json=json_body, request=req)


def _make_row(id_: str, kind: str, content: str, ts: str = "2026-05-08T10:00:00+00:00"):
    return {
        "id": id_,
        "agent_id": "ag_test",
        "kind": kind,
        "content": content,
        "created_at": ts,
    }


def test_append_posts_and_returns_entry():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = request.content.decode()
        return _api_response(201, _make_row("mem_1", "learning", "today learned X"))

    transport = httpx.MockTransport(handler)
    mem = RdsAdapter("ag_test", api_base_url="http://fake")
    mem._client = httpx.Client(base_url="http://fake", transport=transport)

    entry = mem.append("learning", "today learned X")
    assert entry.id == "mem_1"
    assert entry.kind == "learning"
    assert entry.content == "today learned X"
    assert isinstance(entry.created_at, datetime)
    assert captured["method"] == "POST"
    assert "agents/ag_test/memories" in captured["url"]


def test_recent_returns_list_in_order():
    rows = [
        _make_row("mem_2", "event", "newer", "2026-05-08T12:00:00+00:00"),
        _make_row("mem_1", "event", "older", "2026-05-08T10:00:00+00:00"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        # cast-api 返时间倒序 · 我们直接用
        return _api_response(200, rows)

    transport = httpx.MockTransport(handler)
    mem = RdsAdapter("ag_test", api_base_url="http://fake")
    mem._client = httpx.Client(base_url="http://fake", transport=transport)

    out = mem.recent(limit=10)
    assert [m.id for m in out] == ["mem_2", "mem_1"]
    assert out[0].content == "newer"


def test_search_client_side_substring():
    rows = [
        _make_row("m1", "event", "alice 来了"),
        _make_row("m2", "event", "bob 报名"),
        _make_row("m3", "learning", "alice 是 VIP"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _api_response(200, rows)

    transport = httpx.MockTransport(handler)
    mem = RdsAdapter("ag_test", api_base_url="http://fake")
    mem._client = httpx.Client(base_url="http://fake", transport=transport)

    hits = mem.search("alice", limit=5)
    assert {h.id for h in hits} == {"m1", "m3"}

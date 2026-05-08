"""Tools registry · register_tool 装饰器 · call 路由 · missing 异常"""

from __future__ import annotations

import httpx
import pytest

from akong_agent_harness import (
    ToolNotRegisteredError,
    Tools,
    register_tool,
)
from akong_agent_harness.tools import clear_registered_tools, get_registered_tool


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registered_tools()
    yield
    clear_registered_tools()


def test_register_tool_records_implementation():
    @register_tool("cast.post")
    def cast_post(content: str, images: list | None = None) -> dict:
        return {"posted": content, "n_images": len(images or [])}

    impl = get_registered_tool("cast.post")
    assert impl is cast_post
    assert impl(content="hi", images=["a.png"]) == {"posted": "hi", "n_images": 1}


def test_tools_call_routes_to_registered_fn():
    @register_tool("cast.send_dm")
    def send_dm(to_user_id: str, content: str) -> dict:
        return {"to": to_user_id, "msg": content}

    rows = [
        {
            "id": "cast.send_dm",
            "name": "send_dm",
            "description": "私信",
            "params_schema_json": "{}",
            "returns_schema_json": "{}",
            "platform": "cast",
            "scope": "normal",
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=rows, request=request)

    transport = httpx.MockTransport(handler)
    tools = Tools.connect("ag_test", platform="cast", api_base_url="http://fake")
    tools._client = httpx.Client(base_url="http://fake", transport=transport)

    specs = tools.list()
    assert len(specs) == 1
    assert specs[0].id == "cast.send_dm"

    out = tools.call("cast.send_dm", to_user_id="u_1", content="hi")
    assert out == {"to": "u_1", "msg": "hi"}


def test_call_missing_tool_raises_not_registered():
    tools = Tools.connect("ag_test", platform="cast", api_base_url="http://fake")
    with pytest.raises(ToolNotRegisteredError) as exc:
        tools.call("cast.unknown", x=1)
    assert "cast.unknown" in str(exc.value)

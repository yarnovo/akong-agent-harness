"""Tools · agent 能调的"动作" (architecture.md §2.4 + §3.1 + D-3 决策 B)

设计:
- harness 维护一个全局 `_TOOL_IMPL` registry · 平台仓 (cast-platform-tools 等) import 时注册
- Tools.connect 通过 cast-api `/api/agents/{id}/tools` 拉 spec
- Tools.call 路由到 registry 里注册的同进程 Python 函数
- 没注册的 tool_id 调 → ToolNotRegisteredError
- 内置 2 个 harness tool: harness.update_memory / harness.update_self · 由 runtime 注入实现
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import httpx

from .memory import DEFAULT_API_BASE_URL


class ToolNotRegisteredError(Exception):
    """tool_id 没在全局 registry 找到实现"""


class ToolError(Exception):
    """tool registry / call 后端失败"""


# 全局 tool 实现注册表 · 平台仓在 import 时通过 register_tool 装饰器注册
_TOOL_IMPL: dict[str, Callable[..., Any]] = {}


def register_tool(tool_id: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """平台仓注册 tool 实现 · cast-platform-tools 等用。

    用法:
        @register_tool("cast.post")
        def cast_post(content: str, images: list[str] | None = None) -> dict:
            ...
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _TOOL_IMPL[tool_id] = fn
        return fn
    return decorator


def get_registered_tool(tool_id: str) -> Callable[..., Any] | None:
    return _TOOL_IMPL.get(tool_id)


def all_registered_tools() -> dict[str, Callable[..., Any]]:
    return dict(_TOOL_IMPL)


def clear_registered_tools() -> None:
    """测试 hook · 清空 registry"""
    _TOOL_IMPL.clear()


@dataclass
class ToolSpec:
    id: str
    name: str
    description: str
    params_schema: dict[str, Any]
    returns_schema: dict[str, Any]
    platform: str
    scope: str

    @classmethod
    def from_api(cls, row: dict[str, Any]) -> "ToolSpec":
        return cls(
            id=row["id"],
            name=row["name"],
            description=row.get("description", ""),
            params_schema=_safe_json(row.get("params_schema_json", "{}")),
            returns_schema=_safe_json(row.get("returns_schema_json", "{}")),
            platform=row["platform"],
            scope=row.get("scope", "normal"),
        )

    def to_openai_function(self) -> dict[str, Any]:
        """转 OpenAI function-calling tool spec"""
        params = self.params_schema or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": self.id.replace(".", "__"),  # OpenAI tool name 不允许 . · 双下划线编码
                "description": self.description or self.name,
                "parameters": params,
            },
        }


def _safe_json(s: str) -> dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}


@runtime_checkable
class ToolsApi(Protocol):
    """agent 一端只看到这个接口"""

    def list(self) -> list[ToolSpec]: ...
    def call(self, tool_id: str, **kwargs: Any) -> Any: ...


class Tools:
    """tools 注册中心 + agent 当前可调子集"""

    def __init__(
        self,
        agent_id: str,
        platform: str | None = None,
        api_base_url: str | None = None,
        *,
        timeout: float = 10.0,
    ) -> None:
        if not agent_id:
            raise ValueError("agent_id required")
        self.agent_id = agent_id
        self.platform = platform
        self.api_base_url = (api_base_url or os.environ.get("AKONG_API_BASE_URL", DEFAULT_API_BASE_URL)).rstrip("/")
        self._client = httpx.Client(base_url=self.api_base_url, timeout=timeout)
        self._cache: list[ToolSpec] | None = None

    @classmethod
    def connect(
        cls,
        agent_id: str,
        platform: str | None = None,
        *,
        backend: str = "auto",
        api_base_url: str | None = None,
    ) -> "Tools":
        if backend not in {"auto", "rds"}:
            raise ValueError(f"unsupported tools backend: {backend}")
        return cls(agent_id, platform=platform, api_base_url=api_base_url)

    def list(self) -> list[ToolSpec]:
        if self._cache is not None:
            return self._cache
        try:
            resp = self._client.get(f"/api/agents/{self.agent_id}/tools")
        except httpx.HTTPError as e:
            raise ToolError(f"list tools failed: {e}") from e
        if resp.status_code >= 400:
            raise ToolError(f"list tools → {resp.status_code} {resp.text[:200]}")
        rows = resp.json()
        specs = [ToolSpec.from_api(r) for r in rows]
        if self.platform:
            specs = [s for s in specs if s.platform in (self.platform, "global", "harness")]
        self._cache = specs
        return specs

    def refresh(self) -> list[ToolSpec]:
        self._cache = None
        return self.list()

    def call(self, tool_id: str, **kwargs: Any) -> Any:
        impl = _TOOL_IMPL.get(tool_id)
        if impl is None:
            raise ToolNotRegisteredError(
                f"tool '{tool_id}' has no registered implementation · "
                f"platform repo must @register_tool('{tool_id}') before calling"
            )
        return impl(**kwargs)

    def close(self) -> None:
        self._client.close()

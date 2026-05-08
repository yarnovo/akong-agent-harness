"""Session · agent 的"短期对话上下文" (跟 Memory 长期沉淀分开 · 老板 5-7 拍)

session = 一次会话内的多轮 turn (user / assistant / tool 消息) · 跨 FC 实例 sticky 必走 RDS。
memory  = 长期记忆 / 学习 / 关系 / 偏好 (architecture.md §3.1) · 走 cast-api memories endpoint。

实现 2 个:
  - InMemorySession · dict 存 · 测试用 / phone-only 场景
  - RdsSession · 走 cast-api `/api/chat_messages` (上游 P0b 同步在建 · 不存在时抛 SessionUnavailable)

Session 接 `dict` 形态消息 (跟 OpenAI chat schema 对齐 · 跟 LLMClient.chat_completion messages 同源):
  {"role": "user" | "assistant" | "tool" | "system", "content": "...", "tool_calls": [...], "tool_call_id": "..."}
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

import httpx


DEFAULT_API_BASE_URL = "https://api.cast.agentaily.com"


class SessionError(Exception):
    """session 后端失败"""


class SessionUnavailable(SessionError):
    """上游 chat_messages endpoint 不可用 (404 / 503) · runtime 该 fallback 到 InMemorySession 或抛"""


@runtime_checkable
class Session(Protocol):
    """agent 短期对话上下文接口"""

    session_id: str

    async def append(self, message: dict[str, Any]) -> None: ...
    async def load(self) -> list[dict[str, Any]]: ...
    async def clear(self) -> None: ...


# === InMemorySession ===


class InMemorySession:
    """dict 存 · 测试用 · 进程内不持久化"""

    def __init__(self, session_id: str, *, initial: list[dict[str, Any]] | None = None) -> None:
        if not session_id:
            raise ValueError("session_id required")
        self.session_id = session_id
        self._messages: list[dict[str, Any]] = list(initial or [])

    async def append(self, message: dict[str, Any]) -> None:
        if not isinstance(message, dict):
            raise TypeError(f"message must be dict · got {type(message).__name__}")
        if "role" not in message:
            raise ValueError("message must have 'role'")
        self._messages.append(dict(message))

    async def load(self) -> list[dict[str, Any]]:
        return [dict(m) for m in self._messages]

    async def clear(self) -> None:
        self._messages.clear()


# === RdsSession (cast-api chat_messages endpoint · P0b 同步在建) ===


class RdsSession:
    """走 cast-api HTTP · 跨 FC 实例 sticky。

    上游 endpoint (P0b subagent 同步建 · 不存在时本 client 抛 SessionUnavailable):
      POST /api/chat_messages
        body: {session_id, role, content, tool_calls?, tool_call_id?, agent_id?}
        201 → message row · 4xx (404/503) → SessionUnavailable
      GET  /api/chat_messages?session_id=xxx&limit=200
        200 → list[message] · 404 (session 不存在 · 当空)/503 → SessionUnavailable
      DELETE /api/chat_messages?session_id=xxx
        204 → ok · 404 当 already empty · 503 → SessionUnavailable

    返回的 message dict 拿掉 cast-api 自带字段 (id / created_at) · 留 chat schema 字段 (role / content / tool_calls / tool_call_id)。
    """

    def __init__(
        self,
        session_id: str,
        *,
        api_base_url: str | None = None,
        agent_id: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not session_id:
            raise ValueError("session_id required")
        self.session_id = session_id
        self.agent_id = agent_id
        self.api_base_url = (
            api_base_url or os.environ.get("AKONG_API_BASE_URL", DEFAULT_API_BASE_URL)
        ).rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = client or httpx.AsyncClient(
            base_url=self.api_base_url, timeout=timeout, trust_env=False
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.request(method, path, **kwargs)
            except httpx.HTTPError as e:
                last_exc = e
                if attempt >= self._max_retries:
                    break
                continue
            # 503 retry · 404/410 直接抛 unavailable
            if resp.status_code in (404, 410):
                raise SessionUnavailable(
                    f"{method} {path} → {resp.status_code} (chat_messages endpoint 未上线?)"
                )
            if resp.status_code == 503:
                last_exc = SessionUnavailable(f"{method} {path} → 503 (cast-api 暂不可用)")
                if attempt < self._max_retries:
                    continue
                raise last_exc
            if resp.status_code >= 400:
                raise SessionError(f"{method} {path} → {resp.status_code} {resp.text[:200]}")
            return resp
        raise SessionError(
            f"{method} {path} failed after {self._max_retries + 1} attempts: {last_exc}"
        )

    @staticmethod
    def _row_to_message(row: dict[str, Any]) -> dict[str, Any]:
        """cast-api row → chat message dict (拿掉 id / created_at / session_id)"""
        m: dict[str, Any] = {"role": row["role"], "content": row.get("content") or ""}
        tcs = row.get("tool_calls")
        if tcs:
            m["tool_calls"] = tcs
        tcid = row.get("tool_call_id")
        if tcid:
            m["tool_call_id"] = tcid
        return m

    async def append(self, message: dict[str, Any]) -> None:
        if not isinstance(message, dict) or "role" not in message:
            raise ValueError("message must be dict with 'role'")
        body: dict[str, Any] = {
            "session_id": self.session_id,
            "role": message["role"],
            "content": message.get("content") or "",
        }
        if message.get("tool_calls"):
            body["tool_calls"] = message["tool_calls"]
        if message.get("tool_call_id"):
            body["tool_call_id"] = message["tool_call_id"]
        if self.agent_id:
            body["agent_id"] = self.agent_id
        await self._request("POST", "/api/chat_messages", json=body)

    async def load(self) -> list[dict[str, Any]]:
        try:
            resp = await self._request(
                "GET", "/api/chat_messages", params={"session_id": self.session_id, "limit": 500}
            )
        except SessionUnavailable:
            # 404 = session 还没消息 (或 endpoint 没上线) · 上层决定 fallback
            raise
        rows = resp.json()
        return [self._row_to_message(r) for r in rows]

    async def clear(self) -> None:
        try:
            await self._request("DELETE", "/api/chat_messages", params={"session_id": self.session_id})
        except SessionUnavailable:
            # 已空当成 ok
            return

    async def close(self) -> None:
        await self._client.aclose()


# === 工厂 ===


def connect(
    session_id: str,
    *,
    backend: str = "auto",
    api_base_url: str | None = None,
    agent_id: str | None = None,
) -> Session:
    """工厂 · 按 backend 选实现。

    backend:
      - 'auto' → env AKONG_SESSION_BACKEND · 默认 rds
      - 'memory' → InMemorySession
      - 'rds' → RdsSession (cast-api)
    """
    if backend == "auto":
        backend = os.environ.get("AKONG_SESSION_BACKEND", "rds")
    if backend == "memory":
        return InMemorySession(session_id)
    if backend == "rds":
        return RdsSession(session_id, api_base_url=api_base_url, agent_id=agent_id)
    raise ValueError(f"unsupported session backend: {backend}")

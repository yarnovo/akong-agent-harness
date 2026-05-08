"""Memory · agent 的"长期记忆" (architecture.md §3.1)

RdsAdapter 通过 cast-api `/api/agents/{id}/memories` HTTP endpoint 调。
MVP search 用 client-side LIKE · cast-api 还没 vector 后端。
snapshot MVP 不实现 (pass) · 等长记忆压缩需求出现再开。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import httpx
from dateutil.parser import isoparse


DEFAULT_API_BASE_URL = "https://api.cast.agentaily.com"


class MemoryError(Exception):
    """memory 后端失败"""


@dataclass
class MemoryEntry:
    id: str
    kind: str
    content: str
    created_at: datetime
    metadata: dict[str, Any] | None = None
    agent_id: str | None = None

    @classmethod
    def from_api(cls, row: dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=row["id"],
            kind=row["kind"],
            content=row["content"],
            created_at=isoparse(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"],
            agent_id=row.get("agent_id"),
        )


@runtime_checkable
class Memory(Protocol):
    def append(self, kind: str, content: str, metadata: dict[str, Any] | None = None) -> MemoryEntry: ...
    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]: ...
    def recent(self, kind: str | None = None, limit: int = 20) -> list[MemoryEntry]: ...
    def snapshot(self) -> None: ...


class RdsAdapter:
    """通过 cast-api HTTP 调 · 同步 httpx client + 简单 retry。

    cast-api endpoint 见 cast-api/src/cast_api/routers/agents.py:
      POST /api/agents/{id}/memories  · append
      GET  /api/agents/{id}/memories  · list (kind / limit query)
    """

    def __init__(
        self,
        agent_id: str,
        api_base_url: str | None = None,
        *,
        timeout: float = 10.0,
        max_retries: int = 2,
    ) -> None:
        if not agent_id:
            raise ValueError("agent_id required")
        self.agent_id = agent_id
        self.api_base_url = (api_base_url or os.environ.get("AKONG_API_BASE_URL", DEFAULT_API_BASE_URL)).rstrip("/")
        self._client = httpx.Client(base_url=self.api_base_url, timeout=timeout)
        self._max_retries = max_retries

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.request(method, path, **kwargs)
                if resp.status_code >= 500 and attempt < self._max_retries:
                    last_exc = MemoryError(f"{method} {path} → {resp.status_code} {resp.text[:200]}")
                    continue
                if resp.status_code >= 400:
                    raise MemoryError(f"{method} {path} → {resp.status_code} {resp.text[:200]}")
                return resp
            except httpx.HTTPError as e:
                last_exc = e
                if attempt >= self._max_retries:
                    break
        raise MemoryError(f"{method} {path} failed after {self._max_retries + 1} attempts: {last_exc}")

    def append(self, kind: str, content: str, metadata: dict[str, Any] | None = None) -> MemoryEntry:
        # cast-api AgentMemoryCreate 不带 metadata 字段 · 元数据 inline 进 content (后续 schema 加再独立)
        body = {"kind": kind, "content": content}
        resp = self._request("POST", f"/api/agents/{self.agent_id}/memories", json=body)
        entry = MemoryEntry.from_api(resp.json())
        entry.metadata = metadata
        return entry

    def recent(self, kind: str | None = None, limit: int = 20) -> list[MemoryEntry]:
        params: dict[str, Any] = {"limit": limit}
        if kind:
            params["kind"] = kind
        resp = self._request("GET", f"/api/agents/{self.agent_id}/memories", params=params)
        return [MemoryEntry.from_api(r) for r in resp.json()]

    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        # MVP: client-side substring 过滤 · cast-api 还没 vector 后端
        # 拉一批最近的 · 在客户端过滤 · 命中前 N 条
        resp = self._request(
            "GET",
            f"/api/agents/{self.agent_id}/memories",
            params={"limit": 200},
        )
        rows = [MemoryEntry.from_api(r) for r in resp.json()]
        q = query.lower()
        hits = [r for r in rows if q in r.content.lower()]
        return hits[:limit]

    def snapshot(self) -> None:
        # 长记忆压缩 · MVP 不实现
        return None

    def close(self) -> None:
        self._client.close()


def connect(
    agent_id: str,
    *,
    backend: str = "auto",
    api_base_url: str | None = None,
) -> Memory:
    if backend == "auto":
        backend = os.environ.get("AKONG_MEMORY_BACKEND", "rds")
    if backend == "rds":
        return RdsAdapter(agent_id, api_base_url=api_base_url)
    raise ValueError(f"unsupported memory backend: {backend}")

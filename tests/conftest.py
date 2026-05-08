"""测试 fixture · 关掉 host 上的 SOCKS / HTTP_PROXY 让 httpx MockTransport 不撞 socksio。"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _disable_proxy_env(monkeypatch):
    for key in (
        "all_proxy",
        "ALL_PROXY",
        "http_proxy",
        "HTTP_PROXY",
        "https_proxy",
        "HTTPS_PROXY",
    ):
        monkeypatch.delenv(key, raising=False)
    yield

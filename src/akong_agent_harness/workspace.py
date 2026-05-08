"""Workspace · agent 的"硬盘" (architecture.md §3.1)

本地 fs 唯一实现 · root 来自 env 或 connect 参数。
prod 部署时 FC 容器 mount NAS 进 `/mnt/nas/agents` · 同一 class 跑。
大文件 upload (OSS) 留接口扩展点 · MVP 不实现。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol, runtime_checkable


class WorkspaceError(Exception):
    """workspace 操作失败"""


@runtime_checkable
class Workspace(Protocol):
    """agent 的"硬盘" 接口 · §3.1"""

    def write_file(self, path: str, content: str) -> None: ...
    def read_file(self, path: str) -> str: ...
    def list_dir(self, path: str = "") -> list[str]: ...
    def delete(self, path: str) -> None: ...
    def upload(self, local_path: str) -> str: ...


def _expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


class LocalFsAdapter:
    """本地 fs 实现 · 也是 NAS 实现 (传 root=/mnt/nas/agents)。

    每个 agent 隔离到 `<root>/<agent_id>/` 目录 · 不允许逃逸。
    """

    def __init__(self, agent_id: str, root: str | Path | None = None) -> None:
        if not agent_id:
            raise ValueError("agent_id required")
        self.agent_id = agent_id
        root = root or os.environ.get("AKONG_WORKSPACE_ROOT", "~/.akong/agents")
        self._base = _expand(root) / agent_id
        self._base.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        if not path:
            return self._base
        # 禁止绝对路径 + ../ 逃逸
        candidate = (self._base / path).resolve()
        try:
            candidate.relative_to(self._base)
        except ValueError as e:
            raise WorkspaceError(f"path escapes workspace: {path}") from e
        return candidate

    def write_file(self, path: str, content: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"not a file: {path}")
        return target.read_text(encoding="utf-8")

    def list_dir(self, path: str = "") -> list[str]:
        target = self._resolve(path)
        if not target.exists():
            return []
        if not target.is_dir():
            raise WorkspaceError(f"not a dir: {path}")
        return sorted(p.name for p in target.iterdir())

    def delete(self, path: str) -> None:
        target = self._resolve(path)
        if not target.exists():
            return
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    def upload(self, local_path: str) -> str:
        # 大文件走 OSS · 留接口扩展点 · MVP 不实现
        raise NotImplementedError(
            "OssAdapter not wired in MVP · only LocalFsAdapter available"
        )


def connect(agent_id: str, *, backend: str = "auto", root: str | Path | None = None) -> Workspace:
    """工厂 · 根据 env / 参数选 adapter。

    backend:
      - "auto" → 读 AKONG_WORKSPACE_BACKEND env · 默认 localfs
      - "localfs" → LocalFsAdapter
    """
    if backend == "auto":
        backend = os.environ.get("AKONG_WORKSPACE_BACKEND", "localfs")
    if backend == "localfs":
        return LocalFsAdapter(agent_id, root=root)
    raise ValueError(f"unsupported workspace backend: {backend}")

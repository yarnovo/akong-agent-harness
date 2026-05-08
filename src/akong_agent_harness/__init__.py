"""通用 agent harness · LLM 周围的软件基础设施。

参考 cast-agents/docs/architecture.md §2 (6 件套) + §3 (虚拟层 SDK)。

3 个高层接口 (agent 一端只看这些 · 不直接读写 fs / DB / OSS):
  - Workspace · agent 的"硬盘"
  - Memory    · agent 的"长期记忆"
  - Tools     · agent 能调的"动作"

1 个 runtime 入口:
  - tick(agent_id, trigger) → TickResult
"""

from . import workspace as _workspace
from . import memory as _memory
from . import tools as _tools
from .memory import (
    Memory,
    MemoryEntry,
    MemoryError,
    RdsAdapter,
)
from .runtime import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    TickResult,
    Trigger,
    tick,
)
from .tools import (
    ToolError,
    ToolNotRegisteredError,
    ToolSpec,
    Tools,
    ToolsApi,
    register_tool,
)
from .workspace import (
    LocalFsAdapter,
    Workspace,
    WorkspaceError,
)


# 工厂别名 · 让 Workspace.connect / Memory.connect / Tools.connect 三接口形态对齐
class _WorkspaceFactory:
    connect = staticmethod(_workspace.connect)


class _MemoryFactory:
    connect = staticmethod(_memory.connect)


# Workspace / Memory 是 Protocol · attach 工厂方法到模块级名字
# 用法: from akong_agent_harness import Workspace; Workspace.connect("ag_x")
Workspace.connect = staticmethod(_workspace.connect)  # type: ignore[attr-defined]
Memory.connect = staticmethod(_memory.connect)  # type: ignore[attr-defined]


__all__ = [
    # interfaces
    "Workspace",
    "Memory",
    "Tools",
    "ToolsApi",
    # data classes
    "MemoryEntry",
    "ToolSpec",
    "Trigger",
    "TickResult",
    # adapters
    "LocalFsAdapter",
    "RdsAdapter",
    # registration
    "register_tool",
    # runtime
    "tick",
    # errors
    "WorkspaceError",
    "MemoryError",
    "ToolError",
    "ToolNotRegisteredError",
    # constants
    "DEFAULT_LLM_BASE_URL",
    "DEFAULT_LLM_MODEL",
]

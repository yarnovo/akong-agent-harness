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
from . import session as _session
from . import tools as _tools
from .llm import (
    AnthropicClient,
    ChatResponse,
    LLMClient,
    LLMError,
    OpenAICompatibleClient,
    ToolCall as LLMToolCall,
    Usage as LLMUsage,
    connect as connect_llm,
)
from .memory import (
    Memory,
    MemoryEntry,
    MemoryError,
    RdsAdapter,
)
from .runtime import (
    AgentDef,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    RunResult,
    TickResult,
    Trigger,
    run,
    tick,
)
from .session import (
    InMemorySession,
    RdsSession,
    Session,
    SessionError,
    SessionUnavailable,
)
from .skills import (
    Skill,
    SkillError,
    SkillRegistry,
    default_registry as default_skill_registry,
    parse_skill_md,
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


# Workspace / Memory / Session 是 Protocol · attach 工厂方法到模块级名字
# 用法: from akong_agent_harness import Workspace; Workspace.connect("ag_x")
Workspace.connect = staticmethod(_workspace.connect)  # type: ignore[attr-defined]
Memory.connect = staticmethod(_memory.connect)  # type: ignore[attr-defined]
Session.connect = staticmethod(_session.connect)  # type: ignore[attr-defined]


__all__ = [
    # interfaces
    "Workspace",
    "Memory",
    "Session",
    "Tools",
    "ToolsApi",
    "LLMClient",
    # data classes
    "MemoryEntry",
    "ToolSpec",
    "Trigger",
    "TickResult",
    "AgentDef",
    "RunResult",
    "ChatResponse",
    "LLMToolCall",
    "LLMUsage",
    # adapters
    "LocalFsAdapter",
    "RdsAdapter",
    "InMemorySession",
    "RdsSession",
    "OpenAICompatibleClient",
    "AnthropicClient",
    # skills
    "Skill",
    "SkillRegistry",
    "default_skill_registry",
    "parse_skill_md",
    # registration
    "register_tool",
    # runtime
    "tick",
    "run",
    # factories
    "connect_llm",
    # errors
    "WorkspaceError",
    "MemoryError",
    "ToolError",
    "ToolNotRegisteredError",
    "SkillError",
    "SessionError",
    "SessionUnavailable",
    "LLMError",
    # constants
    "DEFAULT_LLM_BASE_URL",
    "DEFAULT_LLM_MODEL",
]

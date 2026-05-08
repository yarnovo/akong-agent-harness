"""akong-agent-harness · v0.2.0 META-PACKAGE (老板 5-9 激进版 8 仓拆 · 兼容层)

⚠️ v0.2.0 起本仓不再 own 实现 · 8 仓拆出 · 本仓只 re-export · 老 import 仍兼容:

    from akong_agent_harness import (...)   # 老用法 · 走兼容 · 本仓 re-export

新代码推荐直接 import 拆出来的子包:

    from akong_llm import LLMClient, OpenAICompatibleClient
    from akong_memory import RdsAdapter
    from akong_session import RdsSession, InMemorySession
    from akong_workspace import LocalFsAdapter
    from akong_tools import register_tool, Tools
    from akong_skills import SkillRegistry, parse_skill_md
    from akong_runtime import run, tick, AgentDef, RunResult
    from akong_pickup import run_with_pickup
"""

# 兼容层: 把老的 sub-module path (akong_agent_harness.llm 等) 也转过去
# 让 `from akong_agent_harness.llm import LLMClient` / `import akong_agent_harness.runtime as r` 仍工作
from akong_llm import (
    AnthropicClient,
    ChatResponse,
    LLMClient,
    LLMError,
    OpenAICompatibleClient,
    ToolCall as LLMToolCall,
    Usage as LLMUsage,
    connect as connect_llm,
)
from akong_memory import (
    Memory,
    MemoryEntry,
    MemoryError,
    RdsAdapter,
)
from akong_runtime import (
    AgentDef,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    RunResult,
    TickResult,
    Trigger,
    run,
    tick,
)
from akong_session import (
    InMemorySession,
    RdsSession,
    Session,
    SessionError,
    SessionUnavailable,
)
from akong_skills import (
    Skill,
    SkillError,
    SkillRegistry,
    default_registry as default_skill_registry,
    parse_skill_md,
)
from akong_tools import (
    ToolError,
    ToolNotRegisteredError,
    ToolSpec,
    Tools,
    ToolsApi,
    register_tool,
)
from akong_workspace import (
    LocalFsAdapter,
    Workspace,
    WorkspaceError,
)


# 老 sub-module path 兼容: akong_agent_harness.{llm,memory,session,workspace,skills,tools,runtime}
# 让 `import akong_agent_harness.runtime as runtime_mod` 仍工作 (test_runtime_smoke 等)
import sys as _sys

# 用 inner module (xxx.xxx 含 httpx 等真实 imports 在文件级) 而不是 package · 让 patch xxx.httpx 仍可工作
import akong_llm.llm as _llm_pkg
import akong_memory.memory as _memory_pkg
import akong_runtime.runtime as _runtime_pkg
import akong_session.session as _session_pkg
import akong_skills.skills as _skills_pkg
import akong_tools.tools as _tools_pkg
import akong_workspace.workspace as _workspace_pkg

_sys.modules[__name__ + ".llm"] = _llm_pkg
_sys.modules[__name__ + ".memory"] = _memory_pkg
_sys.modules[__name__ + ".session"] = _session_pkg
_sys.modules[__name__ + ".skills"] = _skills_pkg
_sys.modules[__name__ + ".tools"] = _tools_pkg
_sys.modules[__name__ + ".runtime"] = _runtime_pkg
_sys.modules[__name__ + ".workspace"] = _workspace_pkg


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

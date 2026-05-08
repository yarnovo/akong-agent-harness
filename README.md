# akong-agent-harness

通用 agent harness · LLM 周围的软件基础设施 · 跨平台 · 跨 fake 平台一致 (cast / B 站 fake / 抖音 fake / ...)。

## v0.2.0 · meta-package (老板 5-9 拍激进版 8 仓拆)

**v0.2.0 起本仓不再 own 实现** · 拆成 8 个独立 GitHub repo · 本仓只 re-export · 老 import 仍兼容:

| 子仓 | 内容 |
|---|---|
| [akong-llm](https://github.com/yarnovo/akong-llm) | LLMClient · OpenAICompatibleClient · AnthropicClient · ChatResponse |
| [akong-memory](https://github.com/yarnovo/akong-memory) | Memory · MemoryEntry · RdsAdapter |
| [akong-session](https://github.com/yarnovo/akong-session) | Session · InMemorySession · RdsSession · SessionUnavailable |
| [akong-workspace](https://github.com/yarnovo/akong-workspace) | Workspace · LocalFsAdapter |
| [akong-skills](https://github.com/yarnovo/akong-skills) | Skill · SkillRegistry · parse_skill_md |
| [akong-tools](https://github.com/yarnovo/akong-tools) | Tools · ToolSpec · register_tool · builtin tools |
| [akong-runtime](https://github.com/yarnovo/akong-runtime) | run · tick · AgentDef · RunResult |
| [akong-pickup](https://github.com/yarnovo/akong-pickup) | run_with_pickup · mid-loop pickup (用户连发新消息自动 inject) |

```python
# 老用法仍兼容 (走本仓 re-export)
from akong_agent_harness import LLMClient, RdsSession, run, AgentDef

# 新用法 · 直接 import 子仓 (减少 transitive 依赖)
from akong_llm import LLMClient
from akong_session import RdsSession
from akong_runtime import run, AgentDef
```



## 6 件套抽象 (架构.md §2 + REQ-001 改造) + 2 件新核心

agent (LLM 一端) 不直接读写 fs / DB / OSS · 走这 8 个高层接口:

- `Workspace` · agent 的"硬盘" (本地 fs / NAS / OSS)
- `Memory` · agent 的"长期记忆" (RDS append-only log · learning / event / preference)
- `Tools` · agent 能调的"动作" (平台 tools registry + 同进程函数注册)
- `Skills` · agent 业务能力包 (SKILL.md frontmatter + body · runtime 装载)
- **`LLMClient`** · LLM provider 抽象 (OpenAI 兼容 + Anthropic native · sync · 5-8 砍 streaming)
- **`Session`** · 短期对话上下文 (跨 FC 实例 sticky · 走 cast-api `chat_messages` RDS · 跟 Memory 长期沉淀分开)
- `runtime.tick(...)` · 老主循环 · sync 单轮 (cast-app /create 链路兼容)
- `runtime.run(...)` · 新主循环 · sync 多轮 tool use loop (借 OpenCode `processGeneration`)

参考: cast-agents 仓 `docs/architecture.md` (§2 6 件套 + §3 虚拟层 SDK) · lead 调研报告 `/Users/yarnb/tmp/agent-framework-research.md`。

## 安装

```bash
uv add akong-agent-harness
```

## 用法 1 · 老 tick (cast-app /create 链路兼容 · 不动)

```python
from akong_agent_harness import Workspace, Memory, Tools, tick, Trigger

ws = Workspace.connect("ag_alice")
mem = Memory.connect("ag_alice", api_base_url="https://api.cast.agentaily.com")
tools = Tools.connect("ag_alice", platform="cast", api_base_url="https://api.cast.agentaily.com")
result = tick("ag_alice", Trigger(kind="manual", payload={"text": "hi"}))
print(result.actions, result.next_wakeup)
```

## 用法 2 · 新 run (多轮 tool use + session 续上下文 · sync · 推荐)

```python
import asyncio
from akong_agent_harness import (
    AgentDef, Session, Tools, Memory, Workspace,
    OpenAICompatibleClient, run,
)

async def main():
    agent = AgentDef(
        id="ag_xiaoyan",
        name="阿空小喜",
        soul="温柔倾听 · 善于挖掘真实需求",
        playbook="先共情 · 再问 · 不打断",
    )
    llm = OpenAICompatibleClient(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="<dashscope-key>",
        model="deepseek-v3.1",
    )
    session = Session.connect(
        "sess_user_42_2026-05-08",
        api_base_url="https://api.cast.agentaily.com",
        agent_id="ag_xiaoyan",
    )
    tools = Tools.connect("ag_xiaoyan", platform="cast")
    memory = Memory.connect("ag_xiaoyan")

    result = await run(
        agent=agent,
        user_message="你好我想咨询",
        session=session,
        llm_client=llm,
        tools=tools,
        memory=memory,
        max_turns=10,
    )
    print(result.final_text, result.stop_reason, result.turns_used)

asyncio.run(main())
```

## LLMClient · 多 provider 抽象

`OpenAICompatibleClient` 通吃: DashScope / DeepSeek / Qwen / 智谱 / 百川 / 本地 vLLM / Together / Groq (业界 90% 兼容协议)。
`AnthropicClient` 单独实现 (tool_use / tool_result 块结构特殊)。

切 provider 只换 client 实例 · `runtime.run` 不动:

```python
# 切 Anthropic
from akong_agent_harness import AnthropicClient
llm = AnthropicClient(api_key="sk-ant-...", model="claude-sonnet-4-5")

# 切 vLLM 自部署
llm = OpenAICompatibleClient(base_url="http://vllm.local:8000/v1", api_key="dummy", model="qwen3-72b")
```

工厂版 (按 env 自动路由):

```python
from akong_agent_harness import connect_llm
llm = connect_llm()  # 读 AKONG_LLM_PROVIDER / AKONG_LLM_MODEL / AKONG_LLM_BASE_URL / AKONG_LLM_API_KEY
```

## Session · 跨 FC 实例 sticky

3 个实现:
- `InMemorySession` · 测试 / phone-only · dict 存
- `RdsSession` · 走 cast-api `/api/chat_messages` · 默认 (跨 FC invoke 续上下文)

跟 `Memory` 区别:
- `Memory` = 长期沉淀 · learning / preference / relationship · LLM 主动写
- `Session` = 短期对话 turn · user / assistant / tool · runtime 自动写

## 配置 (env)

| env | 默认 | 说明 |
|---|---|---|
| `AKONG_WORKSPACE_BACKEND` | `localfs` | 当前只实现 localfs (NAS 复用同 class · 改 root) |
| `AKONG_WORKSPACE_ROOT` | `~/.akong/agents` | workspace 根目录 (prod 配 `/mnt/nas/agents`) |
| `AKONG_MEMORY_BACKEND` | `rds` | 通过 cast-api HTTP 调 |
| `AKONG_SESSION_BACKEND` | `rds` | `rds` (cast-api chat_messages) / `memory` (test) |
| `AKONG_API_BASE_URL` | `https://api.cast.agentaily.com` | cast-api endpoint |
| `AKONG_LLM_PROVIDER` | `openai-compatible` | `openai-compatible` / `anthropic` |
| `AKONG_LLM_API_KEY` | (必配) | provider api key |
| `AKONG_LLM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible endpoint |
| `AKONG_LLM_MODEL` | `deepseek-v3.1` | LLM 模型名 |

## 平台 tool 注册

平台仓 (例 `cast-platform-tools`) import 时注册自己的 tool 函数:

```python
from akong_agent_harness import register_tool

@register_tool("cast.post")
def cast_post(content: str, images: list[str] | None = None) -> dict:
    ...
```

harness 在 `Tools.call(tool_id, **kwargs)` 时查全局 registry 执行。

## 上下游契约

见 `CONTRACTS.md`。

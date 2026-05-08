# CONTRACTS

akong-agent-harness 的上下游契约声明。改本仓任一公开 API · 必同 PR 改本文件 · lead 起 grep CONTRACTS subagent 扫报告。

## 上游 (本仓依赖)

### cast-api (源仓 cast-api · prod `https://api.cast.agentaily.com`)

| Endpoint | 用法 | 调用点 |
|---|---|---|
| `GET /api/agents/{id}` | 拉 agent record (id/name/soul/playbook/style/skills/metadata_json) | `runtime.tick` `_fetch_agent_bundle` |
| `POST /api/agents/{id}/memories` | 写 long-term memory | `memory.RdsAdapter.append` · `runtime` builtin `harness__update_memory` |
| `GET /api/agents/{id}/memories` | 读 recent memory | `memory.RdsAdapter.recent` / `search` |
| `POST /api/agents/{id}/update-self` | self-evolution change_log | `runtime` builtin `harness__update_self` |
| `GET /api/agents/{id}/tools` | 拉 platform tool spec | `tools.Tools.list` |
| **`POST /api/chat_messages`** | session append (P0b 新建 · 老板 5-7 拍 chat_messages 跟 memories 分开) | `session.RdsSession.append` |
| **`GET /api/chat_messages?session_id=xxx`** | session load (跨 FC 实例) | `session.RdsSession.load` |
| **`DELETE /api/chat_messages?session_id=xxx`** | session clear | `session.RdsSession.clear` |

**chat_messages 表 schema (cast-api 侧建 · 本仓只消费)**:

```sql
CREATE TABLE chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(128) NOT NULL,
    agent_id VARCHAR(64),               -- 可空 · 跨 agent session 设计预留
    role VARCHAR(16) NOT NULL,          -- user / assistant / tool / system
    content TEXT,                       -- 文本部分 (可空 · LLM 只调 tool 时为空)
    tool_calls JSONB,                   -- assistant 调 tool 时的 tool_calls list
    tool_call_id VARCHAR(128),          -- role=tool 时关联的 tool_use id
    created_at TIMESTAMPTZ DEFAULT now(),
    INDEX idx_session_id (session_id, created_at)
);
```

**RdsSession 兜底**: cast-api 返 404/410 (endpoint 未上线) 或 503 (临时不可用) → 抛 `SessionUnavailable` · 上层 caller 决定 fallback 到 `InMemorySession` 或硬失败。

### LLM provider (OpenAI-compatible / Anthropic)

| Provider | endpoint | SDK |
|---|---|---|
| DashScope (阿里灵积) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `openai` |
| DeepSeek | `https://api.deepseek.com/v1` | `openai` |
| 智谱 / 百川 / Qwen / vLLM 自部署 | 各自 base_url | `openai` |
| Anthropic | `https://api.anthropic.com` | `anthropic` |

切 provider 只动 `LLMClient` 实例 · `runtime.run` 不动。

## 下游 (依赖本仓的仓)

### cast-agents (源仓 cast-agents · 各 agent 业务实现)

| 用法 | 函数 | 备注 |
|---|---|---|
| 老 tick 单轮 (cast-app /create 链路) | `tick(agent_id, trigger)` | sync · 兼容保留 · 不 streaming · 不 session |
| 新 run 多轮 + session (推荐 · m./chat./api.<agent>.<team> 域名) | `run(agent, user_message, session, llm_client, ...)` | sync return RunResult · max_turns 默认 10 |

### cast-platform-tools (源仓 cast-platform-tools)

| 用法 | 函数 |
|---|---|
| 注册平台 tool 实现 | `@register_tool("cast.post")` |
| harness 调用 | `Tools.call(tool_id, **kwargs)` |

## 公开 API surface (本仓 export · 改一律算 breaking)

```python
from akong_agent_harness import (
    # 老接口 (不动)
    Workspace, Memory, Tools, Skill, SkillRegistry, Trigger, TickResult, tick,
    LocalFsAdapter, RdsAdapter, register_tool,
    # 新接口 (REQ-001)
    LLMClient, OpenAICompatibleClient, AnthropicClient, ChatResponse, connect_llm,
    Session, InMemorySession, RdsSession, SessionUnavailable,
    AgentDef, RunResult, run,
)
```

## 不在本仓负责

- cast-api `chat_messages` table migration (cast-api 维护人 own)
- cast-platform-tools 具体 tool 实现 (cast 平台 maintainer own)
- 各 agent `AgentDef` 的填充 (cast-agents 各 agent 仓 maintainer own)
- 凭证管理 (vault skill own · 本仓只读 env)

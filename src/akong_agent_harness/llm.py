"""LLMClient · LLM provider 抽象 (架构.md §2.7 + lead 调研 /Users/yarnb/tmp/agent-framework-research.md §3.1)

借 Pydantic AI 双层 (Model + Provider) · 但简化 ·

老板 5-8 紧急砍 streaming · `chat_completion` 直接 sync 返完整 `ChatResponse` ·
不 yield · 不 SSE · 不 async generator。

实现 2 个:
  - OpenAICompatibleClient · 走 openai SDK · 通吃 DashScope / DeepSeek / Qwen / 智谱 / 百川 / vLLM (业界 90% 兼容)
  - AnthropicClient · 走 anthropic SDK · tool_use 块结构特殊处理 (Anthropic 不走 OpenAI tool_calls)

LLM provider 调用一律 normalize 到 ChatResponse (统一字段 message/stop_reason/tool_calls/usage)。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# === 事件 / 响应数据类 ===


@dataclass
class ToolCall:
    """LLM 的 tool 调用请求 (统一 OpenAI / Anthropic 形态)

    id        · provider 给的 call id (OpenAI tool_calls[].id · Anthropic content_block tool_use.id)
    name      · function name (harness 传进去的 tool_id 编码后 · runtime 解码)
    arguments · JSON 字符串 (OpenAI 直接是 str · Anthropic 是 dict · 这里统一成 str)
    """

    id: str
    name: str
    arguments: str


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    """LLM 一轮调用的完整结果 (sync · 非 stream)

    role          · "assistant" 固定
    content       · 文本部分 (可空 · LLM 只调 tool 时为 "")
    tool_calls    · 本轮要执行的 tool 调用列表 (空 = 没调)
    stop_reason   · "end_turn" | "tool_use" | "max_tokens" | "stop_sequence" | provider raw
    usage         · token 统计
    raw           · provider 原始返回 (debug 用)
    """

    role: str = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    raw: Any = None

    def to_message_dict(self) -> dict[str, Any]:
        """转成可 append 进 session.messages 的 dict (兼 OpenAI / Anthropic schema 中间表示)"""
        m: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            m["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return m


# === Protocol ===


class LLMError(Exception):
    """LLM provider 调用失败"""


@runtime_checkable
class LLMClient(Protocol):
    """LLM provider 抽象 (sync · 老板 5-8 砍 streaming)

    实现侧 normalize provider response 到 ChatResponse · 不暴露 OpenAI / Anthropic 差异。
    """

    model_id: str

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResponse: ...


# === OpenAI 兼容 (DashScope / DeepSeek / Qwen / 智谱 / vLLM / Together / Groq) ===


class OpenAICompatibleClient:
    """走 openai SDK · base_url 切 provider · 通吃 90% 模型。

    DeepSeek-v3.1 注: thinking mode 跟 function calling 互斥 · model 含 'deepseek' 自动塞 enable_thinking=False。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        client: Any | None = None,  # 测试注入用 (mock OpenAI)
    ) -> None:
        if not api_key and client is None:
            raise ValueError("api_key required (or pass client=)")
        if not model:
            raise ValueError("model required")
        self.base_url = base_url
        self.api_key = api_key
        self.model_id = model
        self.timeout = timeout
        if client is not None:
            self._client = client
        else:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        extra_body: dict[str, Any] = dict(kwargs.pop("extra_body", None) or {})
        if "deepseek" in self.model_id.lower():
            extra_body.setdefault("enable_thinking", False)

        try:
            completion = await self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
                extra_body=extra_body or None,
                **kwargs,
            )
        except Exception as e:
            raise LLMError(f"OpenAI-compatible call failed: {e}") from e

        choice = completion.choices[0]
        msg = choice.message
        raw_tool_calls = getattr(msg, "tool_calls", None) or []
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments or "{}",
            )
            for tc in raw_tool_calls
        ]

        # finish_reason 映射: 'tool_calls' → 'tool_use' · 'stop' → 'end_turn'
        finish = getattr(choice, "finish_reason", None) or ""
        if tool_calls and finish in ("tool_calls", "function_call", ""):
            stop_reason = "tool_use"
        elif finish == "stop":
            stop_reason = "end_turn"
        elif finish == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = finish or "end_turn"

        usage_raw = getattr(completion, "usage", None)
        usage = Usage(
            prompt_tokens=getattr(usage_raw, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_raw, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage_raw, "total_tokens", 0) or 0,
        )

        return ChatResponse(
            role="assistant",
            content=msg.content or "",
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            raw=completion,
        )


# === Anthropic native (claude-* · tool_use / tool_result 块) ===


class AnthropicClient:
    """走 anthropic SDK · messages API · tool_use / tool_result 块结构。

    Anthropic schema 跟 OpenAI 不同:
      - 没 system role (system 单独传)
      - tool_use / tool_result 是 content blocks · 不在 tool_calls 字段
      - tool_result content 必紧邻 tool_use (LLM history 拼接要小心)

    本实现接收 OpenAI 风 messages (有 system role / tool role · tool_calls 字段) · 内部 normalize 到 Anthropic schema。
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        client: Any | None = None,  # 测试注入
    ) -> None:
        if not api_key and client is None:
            raise ValueError("api_key required (or pass client=)")
        if not model:
            raise ValueError("model required")
        self.api_key = api_key
        self.model_id = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        if client is not None:
            self._client = client
        else:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=api_key, timeout=timeout)

    @staticmethod
    def _normalize_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """把 OpenAI 风 messages → (system_prompt, anthropic_messages)

        - role=system 拎出来合成 system_prompt
        - role=tool · tool_call_id=xxx → role=user content=[tool_result block]
        - role=assistant + tool_calls → role=assistant content=[text? tool_use blocks]
        """
        sys_parts: list[str] = []
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                content = m.get("content") or ""
                if content:
                    sys_parts.append(content)
                continue
            if role == "tool":
                tcid = m.get("tool_call_id")
                content = m.get("content") or ""
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tcid,
                                "content": content,
                            }
                        ],
                    }
                )
                continue
            if role == "assistant":
                blocks: list[dict[str, Any]] = []
                text = m.get("content") or ""
                if text:
                    blocks.append({"type": "text", "text": text})
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    args = fn.get("arguments", "{}")
                    try:
                        args_obj = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        args_obj = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": name,
                            "input": args_obj,
                        }
                    )
                out.append({"role": "assistant", "content": blocks or text or ""})
                continue
            # user
            out.append({"role": role or "user", "content": m.get("content") or ""})
        return "\n\n".join(sys_parts), out

    @staticmethod
    def _convert_tools(openai_tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """OpenAI function tool spec → Anthropic tool spec"""
        if not openai_tools:
            return None
        out: list[dict[str, Any]] = []
        for t in openai_tools:
            fn = t.get("function") or {}
            out.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        return out

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        system, anthropic_msgs = self._normalize_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        try:
            resp = await self._client.messages.create(
                model=self.model_id,
                max_tokens=kwargs.pop("max_tokens", self.max_tokens),
                system=system or None,
                messages=anthropic_msgs,
                tools=anthropic_tools,
                **kwargs,
            )
        except Exception as e:
            raise LLMError(f"Anthropic call failed: {e}") from e

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                args = getattr(block, "input", {}) or {}
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=json.dumps(args, ensure_ascii=False),
                    )
                )

        # Anthropic stop_reason: end_turn / tool_use / max_tokens / stop_sequence
        stop_reason = getattr(resp, "stop_reason", None) or "end_turn"

        usage_raw = getattr(resp, "usage", None)
        usage = Usage(
            prompt_tokens=getattr(usage_raw, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage_raw, "output_tokens", 0) or 0,
            total_tokens=(
                (getattr(usage_raw, "input_tokens", 0) or 0)
                + (getattr(usage_raw, "output_tokens", 0) or 0)
            ),
        )

        return ChatResponse(
            role="assistant",
            content="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            raw=resp,
        )


# === 工厂 ===


def connect(
    *,
    provider: str = "auto",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """工厂 · 按 provider 字符串选 client。

    provider:
      - 'openai' / 'openai-compatible' → OpenAICompatibleClient
      - 'anthropic' → AnthropicClient
      - 'auto' → env AKONG_LLM_PROVIDER · 默认 openai-compatible
    """
    if provider == "auto":
        provider = os.environ.get("AKONG_LLM_PROVIDER", "openai-compatible")
    model = model or os.environ.get("AKONG_LLM_MODEL", "deepseek-v3.1")
    api_key = api_key or os.environ.get("AKONG_LLM_API_KEY", "")

    if provider in ("openai", "openai-compatible"):
        base_url = base_url or os.environ.get(
            "AKONG_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        return OpenAICompatibleClient(base_url=base_url, api_key=api_key, model=model)
    if provider == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)
    raise ValueError(f"unsupported llm provider: {provider}")

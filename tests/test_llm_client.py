"""LLMClient 单测 · OpenAICompatibleClient + AnthropicClient · normalize 到 ChatResponse"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from akong_agent_harness.llm import (
    AnthropicClient,
    ChatResponse,
    LLMError,
    OpenAICompatibleClient,
    ToolCall,
    Usage,
)


# === OpenAICompatibleClient mocks ===


@dataclass
class _OAIFn:
    name: str
    arguments: str


@dataclass
class _OAITC:
    id: str
    function: _OAIFn
    type: str = "function"


class _OAIMessage:
    def __init__(self, content: str = "", tool_calls: list[_OAITC] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or None


class _OAIChoice:
    def __init__(self, message: _OAIMessage, finish_reason: str = "stop") -> None:
        self.message = message
        self.finish_reason = finish_reason


@dataclass
class _OAIUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5
    total_tokens: int = 15


class _OAICompletion:
    def __init__(self, choices: list[_OAIChoice], usage: _OAIUsage | None = None) -> None:
        self.choices = choices
        self.usage = usage or _OAIUsage()


class _FakeAsyncOpenAI:
    def __init__(self, scripted: list[_OAICompletion]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []
        self.chat = self  # type: ignore

    @property
    def completions(self) -> "_FakeAsyncOpenAI":
        return self

    async def create(self, **kwargs: Any) -> _OAICompletion:
        self.calls.append(kwargs)
        if not self._scripted:
            raise RuntimeError("no more scripted")
        return self._scripted.pop(0)


async def test_openai_compatible_text_only():
    fake = _FakeAsyncOpenAI(
        [_OAICompletion([_OAIChoice(_OAIMessage(content="hello world"), finish_reason="stop")])]
    )
    cli = OpenAICompatibleClient(
        base_url="http://fake", api_key="x", model="qwen2.5", client=fake
    )
    resp = await cli.chat_completion([{"role": "user", "content": "hi"}])
    assert isinstance(resp, ChatResponse)
    assert resp.content == "hello world"
    assert resp.tool_calls == []
    assert resp.stop_reason == "end_turn"
    assert resp.usage.total_tokens == 15


async def test_openai_compatible_tool_call():
    fake = _FakeAsyncOpenAI(
        [
            _OAICompletion(
                [
                    _OAIChoice(
                        _OAIMessage(
                            content="",
                            tool_calls=[
                                _OAITC(
                                    id="call_a",
                                    function=_OAIFn(
                                        name="cast__send_dm",
                                        arguments=json.dumps({"to": "u_b", "msg": "hi"}),
                                    ),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            )
        ]
    )
    cli = OpenAICompatibleClient(
        base_url="http://fake", api_key="x", model="qwen2.5", client=fake
    )
    resp = await cli.chat_completion(
        [{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "cast__send_dm",
                    "description": "dm",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "cast__send_dm"
    assert resp.tool_calls[0].id == "call_a"
    args = json.loads(resp.tool_calls[0].arguments)
    assert args["to"] == "u_b"


async def test_openai_compatible_deepseek_thinking_disabled():
    """DeepSeek model 自动塞 enable_thinking=False"""
    fake = _FakeAsyncOpenAI(
        [_OAICompletion([_OAIChoice(_OAIMessage(content="ok"), finish_reason="stop")])]
    )
    cli = OpenAICompatibleClient(
        base_url="http://fake", api_key="x", model="deepseek-v3.1", client=fake
    )
    await cli.chat_completion([{"role": "user", "content": "hi"}])
    assert fake.calls[0]["extra_body"] == {"enable_thinking": False}


async def test_openai_compatible_error_wrapped():
    class _Boom:
        chat = None

        @property
        def completions(self):
            return self

        async def create(self, **kwargs):
            raise RuntimeError("upstream 500")

    boom = _Boom()
    boom.chat = boom  # type: ignore
    cli = OpenAICompatibleClient(base_url="http://fake", api_key="x", model="m", client=boom)
    with pytest.raises(LLMError) as ei:
        await cli.chat_completion([{"role": "user", "content": "hi"}])
    assert "upstream 500" in str(ei.value)


# === AnthropicClient mocks ===


@dataclass
class _AntTextBlock:
    text: str
    type: str = "text"


@dataclass
class _AntToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class _AntUsage:
    input_tokens: int = 20
    output_tokens: int = 7


class _AntResponse:
    def __init__(self, content: list[Any], stop_reason: str = "end_turn") -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = _AntUsage()


class _FakeAsyncAnthropic:
    def __init__(self, scripted: list[_AntResponse]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []
        self.messages = self  # type: ignore

    async def create(self, **kwargs: Any) -> _AntResponse:
        self.calls.append(kwargs)
        return self._scripted.pop(0)


async def test_anthropic_text_only():
    fake = _FakeAsyncAnthropic([_AntResponse(content=[_AntTextBlock(text="hi from claude")])])
    cli = AnthropicClient(api_key="x", model="claude-sonnet-4-5", client=fake)
    resp = await cli.chat_completion([{"role": "user", "content": "hello"}])
    assert resp.content == "hi from claude"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.prompt_tokens == 20
    assert resp.usage.completion_tokens == 7


async def test_anthropic_tool_use():
    fake = _FakeAsyncAnthropic(
        [
            _AntResponse(
                content=[
                    _AntTextBlock(text="let me call tool"),
                    _AntToolUseBlock(id="tu_1", name="cast__post", input={"text": "hi"}),
                ],
                stop_reason="tool_use",
            )
        ]
    )
    cli = AnthropicClient(api_key="x", model="claude-sonnet-4-5", client=fake)
    resp = await cli.chat_completion(
        [{"role": "user", "content": "post"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "cast__post",
                    "description": "post",
                    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
                },
            }
        ],
    )
    assert resp.stop_reason == "tool_use"
    assert resp.content == "let me call tool"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "cast__post"
    assert resp.tool_calls[0].id == "tu_1"
    assert json.loads(resp.tool_calls[0].arguments) == {"text": "hi"}
    # 验证 tool spec 正确转换 (input_schema 字段)
    sent_tools = fake.calls[0]["tools"]
    assert sent_tools[0]["name"] == "cast__post"
    assert "input_schema" in sent_tools[0]


async def test_anthropic_normalize_tool_result_message():
    """OpenAI 风 'role=tool' 消息 → Anthropic role=user content=[tool_result block]"""
    sys, msgs = AnthropicClient._normalize_messages(
        [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": json.dumps({"x": 1})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": '{"ok": true}'},
        ]
    )
    assert sys == "you are helpful"
    assert msgs[0] == {"role": "user", "content": "go"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0]["type"] == "tool_use"
    assert msgs[1]["content"][0]["id"] == "tc_1"
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][0]["tool_use_id"] == "tc_1"

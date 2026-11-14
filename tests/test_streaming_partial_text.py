"""Streamed replies must reach the client as the model writes them.

The SDK yields each assistant turn only once it is complete unless partial
messages are switched on, so clients such as Open WebUI showed the whole
reply at once. The text deltas of the StreamEvents are now forwarded, and
the complete AssistantMessage that follows them is not sent again.
"""

import asyncio
import json

from openai.types.chat import ChatCompletionChunk

from src import main as main_mod
from src.models import ChatCompletionRequest, Message
from tests.test_streaming_tool_calls import TOOLS


def _delta(text, parent_tool_use_id=None):
    return {
        "uuid": "u",
        "session_id": "s",
        "parent_tool_use_id": parent_tool_use_id,
        "event": {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
    }


def _event(event_type):
    return {
        "uuid": "u",
        "session_id": "s",
        "parent_tool_use_id": None,
        "event": {"type": event_type},
    }


def _stream(monkeypatch, sdk_chunks, **request_kwargs):
    async def fake_run_completion(**kwargs):
        for chunk in sdk_chunks:
            yield chunk

    monkeypatch.setattr(main_mod.claude_cli, "run_completion", fake_run_completion)
    request = ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[Message(role="user", content="hi")],
        stream=True,
        **request_kwargs,
    )

    async def collect():
        return [c async for c in main_mod.generate_streaming_response(request, "req-pt")]

    chunks = []
    for line in "".join(asyncio.run(collect())).splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(ChatCompletionChunk.model_validate(json.loads(line[6:])))
    return chunks


def _contents(chunks):
    return [ch.delta.content for c in chunks for ch in c.choices if ch.delta.content]


def test_text_deltas_are_forwarded_one_by_one_and_not_repeated(monkeypatch):
    chunks = _stream(
        monkeypatch,
        [
            _event("message_start"),
            _delta("Hello"),
            _delta(", "),
            _delta("world.\n\n"),
            _delta("Bye."),
            _event("content_block_stop"),
            {"content": [{"type": "text", "text": "Hello, world.\n\nBye."}]},
            {"subtype": "success", "result": "Hello, world.\n\nBye."},
        ],
    )

    assert _contents(chunks) == ["Hello", ", ", "world.\n\n", "Bye."]
    roles = [ch.delta.role for c in chunks for ch in c.choices if ch.delta.role]
    assert roles == ["assistant"]
    finish = [ch.finish_reason for c in chunks for ch in c.choices if ch.finish_reason]
    assert finish == ["stop"]


def test_later_turn_without_deltas_is_still_sent(monkeypatch):
    chunks = _stream(
        monkeypatch,
        [
            _delta("First."),
            {"content": [{"type": "text", "text": "First."}]},
            {"content": [{"type": "text", "text": "Second."}]},
        ],
    )

    assert _contents(chunks) == ["First.", "Second."]


def test_subagent_deltas_are_not_forwarded(monkeypatch):
    chunks = _stream(
        monkeypatch,
        [
            _delta("internal", parent_tool_use_id="toolu_1"),
            {"content": [{"type": "text", "text": "Answer."}]},
        ],
    )

    assert _contents(chunks) == ["Answer."]


def _tool_deltas(chunks):
    return [tc for c in chunks for ch in c.choices for tc in (ch.delta.tool_calls or [])]


def _finish(chunks):
    return [ch.finish_reason for c in chunks for ch in c.choices if ch.finish_reason]


def _split(text, size=3):
    return [_delta(text[i : i + size]) for i in range(0, len(text), size)]


def test_with_client_tools_a_split_tool_call_block_is_parsed(monkeypatch):
    text = '```tool_calls\n[{"name": "get_weather", "arguments": {"city": "Berlin"}}]\n```'
    chunks = _stream(
        monkeypatch,
        [*_split(text), {"content": [{"type": "text", "text": text}]}],
        tools=TOOLS,
    )

    assert [tc.function.name for tc in _tool_deltas(chunks)] == ["get_weather"]
    assert not any("`" in t or "get_weather" in t for t in _contents(chunks))
    assert _finish(chunks) == ["tool_calls"]


def test_with_client_tools_plain_text_streams(monkeypatch):
    text = "Berlin is sunny [mostly] and <b>warm</b>."
    chunks = _stream(
        monkeypatch,
        [*_split(text), {"content": [{"type": "text", "text": text}]}],
        tools=TOOLS,
    )

    contents = _contents(chunks)
    assert len(contents) > 5
    assert "".join(contents) == text
    assert not _tool_deltas(chunks)
    assert _finish(chunks) == ["stop"]


def test_with_client_tools_text_before_a_call_streams_and_markup_is_held(monkeypatch):
    text = (
        "Let me check the weather.\n\n"
        '<invoke name="get_weather">\n<parameter name="city">Berlin</parameter>\n</invoke>'
    )
    chunks = _stream(monkeypatch, _split(text), tools=TOOLS)

    assert "".join(_contents(chunks)) == "Let me check the weather.\n\n"
    assert [tc.function.name for tc in _tool_deltas(chunks)] == ["get_weather"]
    assert _finish(chunks) == ["tool_calls"]


def test_with_client_tools_markup_for_unknown_tool_is_sent_as_text(monkeypatch):
    text = 'Try this:\n<invoke name="Bash">\n<parameter name="command">ls</parameter>\n</invoke>'
    chunks = _stream(monkeypatch, _split(text), tools=TOOLS)

    assert "".join(_contents(chunks)) == text
    assert not _tool_deltas(chunks)
    assert _finish(chunks) == ["stop"]


def test_partial_messages_are_requested_only_when_streaming(monkeypatch):
    seen = []

    async def fake_query(prompt, options):
        seen.append(options.include_partial_messages)
        return
        yield

    monkeypatch.setattr("src.claude_cli.query", fake_query)
    cli = main_mod.claude_cli

    async def drain(stream):
        async for _ in cli.run_completion(prompt="hi", stream=stream):
            pass

    asyncio.run(drain(True))
    asyncio.run(drain(False))

    assert seen == [True, False]

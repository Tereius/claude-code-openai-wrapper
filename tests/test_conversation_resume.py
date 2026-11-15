"""A conversation's next turn resumes the CLI session that answered the last.

Flattened into one prompt, the history never hits the prompt cache, so each
turn paid to write the whole conversation to the cache again. Resuming the
session sends only the new messages and reads the history from the cache.
"""

import asyncio
import tempfile
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src import main as main_mod
from src.conversation_index import ConversationIndex
from src.models import ChatCompletionRequest, FunctionCall, Message, ToolCall
from tests.test_streaming_tool_calls import TOOLS


def _user(text):
    return Message(role="user", content=text)


def _assistant(text=None, tool_calls=None):
    return Message(role="assistant", content=text, tool_calls=tool_calls)


def _call(call_id, name, arguments):
    return ToolCall(id=call_id, function=FunctionCall(name=name, arguments=arguments))


class TestConversationIndex:
    def test_next_turn_resumes_the_session_that_produced_the_reply(self):
        index = ConversationIndex()
        first = [Message(role="system", content="Be brief."), _user("Hi")]
        index.record(first, _assistant("Hello!"), "cli-1")

        plan = index.lookup(first + [_assistant("Hello!"), _user("How are you?")])

        assert plan.session_id == "cli-1"
        assert [m.content for m in plan.new_messages] == ["How are you?"]

    def test_whitespace_the_client_changed_in_the_reply_still_matches(self):
        index = ConversationIndex()
        index.record([_user("Hi")], _assistant("Hello!\n\nHow can I help?\n"), "cli-1")

        plan = index.lookup([_user("Hi"), _assistant("Hello! How can I help?"), _user("x")])

        assert plan.session_id == "cli-1"

    def test_edited_history_or_reply_does_not_match(self):
        index = ConversationIndex()
        index.record([_user("Hi")], _assistant("Hello!"), "cli-1")

        assert index.lookup([_user("Hey"), _assistant("Hello!"), _user("x")]) is None
        assert index.lookup([_user("Hi"), _assistant("Howdy!"), _user("x")]) is None

    def test_tool_call_ids_and_argument_formatting_are_ignored(self):
        index = ConversationIndex()
        index.record([_user("Weather?")], _assistant(tool_calls=[_call("a", "w", '{"c":1}')]), "s")

        echoed = _assistant(tool_calls=[_call("b", "w", '{ "c": 1 }')])
        tool_result = Message(role="tool", content="sunny", tool_call_id="b")
        plan = index.lookup([_user("Weather?"), echoed, tool_result])

        assert plan.session_id == "s"
        assert plan.new_messages == [tool_result]

    def test_no_resume_without_new_user_messages(self):
        index = ConversationIndex()
        index.record([_user("Hi")], _assistant("Hello!"), "cli-1")

        assert index.lookup([_user("Hi"), _assistant("Hello!")]) is None
        assert index.lookup([_user("Hi")]) is None

    def test_tool_calls_the_client_ran_itself_follow_the_recorded_reply(self):
        """Open WebUI runs some tools itself and inserts call and result."""
        index = ConversationIndex()
        client_call = _assistant(tool_calls=[_call("c1", "get_time", "{}")])
        first = [_user("Write a poem"), client_call, Message(role="tool", content="20:53")]
        index.record(first, _assistant("Roses..."), "cli-1")

        second_call = _assistant(tool_calls=[_call("c2", "get_time", "{}")])
        new = [_user("Another one"), second_call, Message(role="tool", content="20:54")]
        plan = index.lookup(first + [_assistant("Roses...")] + new)

        assert plan.session_id == "cli-1"
        assert plan.new_messages == new

    def test_unrecorded_reply_is_resent_after_the_latest_recorded_one(self):
        index = ConversationIndex()
        index.record([_user("a")], _assistant("A"), "cli-1")

        tail = [_user("b"), _assistant("B, never recorded"), _user("c")]
        plan = index.lookup([_user("a"), _assistant("A")] + tail)

        assert plan.session_id == "cli-1"
        assert plan.new_messages == tail

    def test_no_resume_when_the_last_message_is_a_reply(self):
        index = ConversationIndex()
        index.record([_user("a")], _assistant("A"), "cli-1")

        assert index.lookup([_user("a"), _assistant("A"), _user("b"), _assistant("B")]) is None

    def test_log_shows_which_message_the_client_changed(self, caplog):
        index = ConversationIndex()
        with caplog.at_level("INFO", logger="src.conversation_index"):
            index.record([_user("Hi")], _assistant("Hello!"), "cli-1")
            index.lookup([_user("Hi"), _assistant("Hello, edited!"), _user("x")])

        recorded, missed = [r.getMessage() for r in caplog.records]
        recorded_parts = recorded.split("[")[1].rstrip("]").split()
        missed_parts = missed.split("[")[1].rstrip("]").split()
        assert recorded.startswith("Recorded CLI session cli-1")
        assert missed.startswith("No CLI session to resume")
        assert recorded_parts[0] == missed_parts[0]
        assert recorded_parts[1] != missed_parts[1]
        assert missed_parts[1].startswith("assistant:") and missed_parts[1].endswith("/14")

    def test_disabled_index_never_resumes(self):
        index = ConversationIndex(enabled=False)
        index.record([_user("Hi")], _assistant("Hello!"), "cli-1")

        assert index.lookup([_user("Hi"), _assistant("Hello!"), _user("x")]) is None

    def test_expired_and_evicted_entries_are_gone(self):
        index = ConversationIndex(max_entries=1, ttl_seconds=60)
        index.record([_user("a")], _assistant("A"), "s-a")
        index.record([_user("b")], _assistant("B"), "s-b")
        assert index.lookup([_user("a"), _assistant("A"), _user("x")]) is None

        with patch("src.conversation_index.time.monotonic", return_value=time.monotonic() + 61):
            assert index.lookup([_user("b"), _assistant("B"), _user("x")]) is None


@pytest.fixture
def fresh_index(monkeypatch):
    # Another test may have left the CLI marked unauthenticated.
    monkeypatch.setattr(main_mod._auth, "cli_health", main_mod._auth.CliHealth())
    index = ConversationIndex()
    monkeypatch.setattr(main_mod, "conversation_index", index)
    return index


def _fake_sdk(monkeypatch, replies):
    """Answer each run_completion call with the next (session id, text) pair."""
    calls = []
    replies = iter(replies)

    async def fake_run_completion(**kwargs):
        calls.append(kwargs)
        session_id, text = next(replies)
        yield {"subtype": "init", "data": {"session_id": session_id, "model": "m"}}
        yield {"content": [{"type": "text", "text": text}]}
        yield {"subtype": "success", "result": text, "session_id": session_id}

    monkeypatch.setattr(main_mod.claude_cli, "run_completion", fake_run_completion)
    return calls


def _stream(messages, **kwargs):
    request = ChatCompletionRequest(
        model="claude-sonnet-4-6", messages=messages, stream=True, **kwargs
    )

    async def collect():
        return [c async for c in main_mod.generate_streaming_response(request, "req")]

    return "".join(asyncio.run(collect()))


def test_streaming_second_turn_sends_only_the_new_message(monkeypatch, fresh_index):
    calls = _fake_sdk(monkeypatch, [("cli-1", "Paris."), ("cli-2", "About 2 million.")])
    history = [_user("Capital of France?")]

    _stream(history)
    _stream(history + [_assistant("Paris."), _user("Population?")])

    assert "session_id" not in calls[0]
    assert calls[1]["session_id"] == "cli-1"
    assert calls[1]["fork_session"] is True
    assert calls[1]["prompt"] == "Population?"
    assert "Human: Capital of France?" in calls[1]["fallback_prompt"]


def test_streaming_tool_call_turn_resumes_with_the_tool_result(monkeypatch, fresh_index):
    call_text = '```tool_calls\n[{"name": "get_weather", "arguments": {"city": "Oslo"}}]\n```'
    calls = _fake_sdk(monkeypatch, [("cli-1", call_text), ("cli-2", "It is sunny.")])
    history = [_user("Weather in Oslo?")]

    sse = _stream(history, tools=TOOLS)
    call_id = sse.split('"tool_calls":[{"index":0,"id":"', 1)[1].split('"')[0]
    _stream(
        history
        + [
            _assistant(tool_calls=[_call(call_id, "get_weather", '{"city": "Oslo"}')]),
            Message(role="tool", content="sunny", tool_call_id=call_id, name="get_weather"),
        ],
        tools=TOOLS,
    )

    assert calls[1]["session_id"] == "cli-1"
    assert calls[1]["prompt"] == f"[Result of get_weather ({call_id}): sunny]"


def test_streaming_turn_after_a_client_run_tool_sends_call_and_result(monkeypatch, fresh_index):
    calls = _fake_sdk(monkeypatch, [("cli-1", "Roses are red."), ("cli-2", "Violets...")])

    def client_tool(call_id, result):
        return [
            _assistant(tool_calls=[_call(call_id, "get_weather", '{"city": "Oslo"}')]),
            Message(role="tool", content=result, tool_call_id=call_id, name="get_weather"),
        ]

    first = [_user("Write a poem")] + client_tool("c1", "sunny")
    _stream(first, tools=TOOLS)
    _stream(
        first + [_assistant("Roses are red."), _user("Another one")] + client_tool("c2", "rainy"),
        tools=TOOLS,
    )

    assert calls[1]["session_id"] == "cli-1"
    assert calls[1]["prompt"] == (
        "Human: Another one\n\n"
        'Assistant: [Called get_weather with arguments: {"city": "Oslo"}]\n\n'
        "Human: [Result of get_weather (c2): rainy]"
    )


def test_streaming_error_reply_is_not_recorded(monkeypatch, fresh_index):
    async def failing_run_completion(**kwargs):
        yield {"subtype": "init", "data": {"session_id": "cli-1", "model": "m"}}
        yield {"subtype": "error_during_execution", "is_error": True, "session_id": "cli-1"}

    monkeypatch.setattr(main_mod.claude_cli, "run_completion", failing_run_completion)
    _stream([_user("Hi")])

    assert fresh_index._entries == {}


def test_non_streaming_second_turn_resumes(monkeypatch, fresh_index):
    calls = _fake_sdk(monkeypatch, [("cli-1", "Paris."), ("cli-2", "About 2 million.")])
    client = TestClient(main_mod.app)
    history = [{"role": "user", "content": "Capital of France?"}]

    first = client.post(
        "/v1/chat/completions", json={"model": "claude-sonnet-4-6", "messages": history}
    )
    reply = first.json()["choices"][0]["message"]
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-sonnet-4-6",
            "messages": history + [reply, {"role": "user", "content": "Population?"}],
        },
    )

    assert first.status_code == second.status_code == 200
    assert calls[1]["session_id"] == "cli-1"
    assert calls[1]["prompt"] == "Population?"


class TestResumeFallback:
    @pytest.fixture
    def cli(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("src.auth.validate_claude_code_auth") as mock_validate:
                with patch("src.auth.auth_manager") as mock_auth:
                    mock_validate.return_value = (True, {"method": "anthropic"})
                    mock_auth.get_claude_code_env_vars.return_value = {}
                    from src.claude_cli import ClaudeCodeCLI

                    yield ClaudeCodeCLI(cwd=temp_dir)

    @pytest.mark.asyncio
    async def test_missing_session_falls_back_to_the_full_prompt(self, cli):
        seen = []

        async def mock_query(prompt, options):
            seen.append((prompt, options.resume, options.fork_session))
            if options.resume:
                raise RuntimeError("No conversation found with session ID: gone")
            yield {"subtype": "success", "result": "ok"}

        with patch("src.claude_cli.query", mock_query):
            chunks = [
                c
                async for c in cli.run_completion(
                    prompt="new message",
                    session_id="gone",
                    fork_session=True,
                    fallback_prompt="Human: everything",
                )
            ]

        assert seen == [("new message", "gone", True), ("Human: everything", None, False)]
        assert chunks[-1]["result"] == "ok"

    @pytest.mark.asyncio
    async def test_plain_chat_ignores_user_and_claude_ai_mcp_servers(self, cli):
        seen = []

        async def mock_query(prompt, options):
            seen.append(options)
            yield {"subtype": "success", "result": "ok"}

        with patch("src.claude_cli.query", mock_query):
            async for _ in cli.run_completion(prompt="hi", disallowed_tools=main_mod.CLAUDE_TOOLS):
                pass

        assert seen[0].strict_mcp_config is True

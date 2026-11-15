"""Responses report the token usage the CLI measured, not a guess.

Usage used to be estimated from the prompt text at four characters per token.
That missed the system prompt and everything the CLI adds, so a request that
sent 23K tokens reported 6.5K, and cached tokens were never shown.
"""

import asyncio
import json
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src import main as main_mod
from src.conversation_index import ConversationIndex
from src.cost_tracker import CostTracker, UsageRecord
from src.models import ChatCompletionRequest, Message, StreamOptions

SDK_USAGE = {
    "input_tokens": 4,
    "cache_creation_input_tokens": 336,
    "cache_read_input_tokens": 11040,
    "output_tokens": 23,
    "server_tool_use": {"web_search_requests": 1},
}


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    # Another test may have left the CLI marked unauthenticated.
    monkeypatch.setattr(main_mod._auth, "cli_health", main_mod._auth.CliHealth())
    monkeypatch.setattr(main_mod, "conversation_index", ConversationIndex())
    tracker = CostTracker()
    monkeypatch.setattr(main_mod, "cost_tracker", tracker)
    return tracker


def _fake_sdk(monkeypatch, usage):
    async def fake_run_completion(**kwargs):
        yield {"subtype": "init", "data": {"session_id": "cli-1", "model": "m"}}
        yield {"content": [{"type": "text", "text": "Paris."}]}
        # Shape of a converted ResultMessage.
        yield {
            "subtype": "success",
            "result": "Paris.",
            "session_id": "cli-1",
            "total_cost_usd": 0.01,
            "num_turns": 1,
            "duration_ms": 1,
            "usage": usage,
        }

    monkeypatch.setattr(main_mod.claude_cli, "run_completion", fake_run_completion)


def test_usage_record_from_sdk_result():
    record = UsageRecord.from_sdk(SDK_USAGE)

    assert record == UsageRecord(
        input_tokens=4,
        output_tokens=23,
        cache_read_tokens=11040,
        cache_creation_tokens=336,
        web_search_requests=1,
    )


def test_non_streaming_reports_cli_usage_with_cached_tokens(monkeypatch, isolated_state):
    _fake_sdk(monkeypatch, SDK_USAGE)

    response = TestClient(main_mod.app).post(
        "/v1/chat/completions",
        json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "Capital?"}]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["usage"] == {
        "prompt_tokens": 4 + 336 + 11040,
        "completion_tokens": 23,
        "total_tokens": 4 + 336 + 11040 + 23,
        "prompt_tokens_details": {"cached_tokens": 11040},
    }
    session = next(iter(isolated_state._sessions.values()))
    assert session.total_cache_read_tokens == 11040
    assert session.total_cache_creation_tokens == 336


def test_non_streaming_estimates_when_the_cli_reports_no_usage(monkeypatch):
    _fake_sdk(monkeypatch, None)

    response = TestClient(main_mod.app).post(
        "/v1/chat/completions",
        json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "Capital?"}]},
    )

    assert response.status_code == 200, response.text
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["prompt_tokens_details"] == {"cached_tokens": 0}


def _stream(stream_options=None):
    request = ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[Message(role="user", content="Capital?")],
        stream=True,
        stream_options=stream_options,
    )

    async def collect():
        return [c async for c in main_mod.generate_streaming_response(request, "req")]

    events = [
        json.loads(line[6:])
        for line in "".join(asyncio.run(collect())).splitlines()
        if line.startswith("data: {")
    ]
    return events[-1]


def test_streaming_reports_cli_usage_in_the_final_chunk(monkeypatch):
    _fake_sdk(monkeypatch, SDK_USAGE)

    final = _stream(StreamOptions(include_usage=True))

    assert final["usage"]["prompt_tokens"] == 4 + 336 + 11040
    assert final["usage"]["prompt_tokens_details"] == {"cached_tokens": 11040}


def test_streaming_records_cost_without_include_usage(monkeypatch, isolated_state):
    _fake_sdk(monkeypatch, SDK_USAGE)

    final = _stream()

    assert final["usage"] is None
    session = next(iter(isolated_state._sessions.values()))
    assert session.total_cache_read_tokens == 11040


def test_anthropic_messages_reports_cache_tokens(monkeypatch):
    _fake_sdk(monkeypatch, SDK_USAGE)

    response = TestClient(main_mod.app).post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Capital?"}],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["usage"] == {
        "input_tokens": 4,
        "output_tokens": 23,
        "cache_creation_input_tokens": 336,
        "cache_read_input_tokens": 11040,
    }


@pytest.mark.asyncio
async def test_auth_probe_sends_a_minimal_request():
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("src.auth.validate_claude_code_auth") as mock_validate:
            with patch("src.auth.auth_manager") as mock_auth:
                mock_validate.return_value = (True, {"method": "anthropic"})
                mock_auth.get_claude_code_env_vars.return_value = {}
                from src.claude_cli import ClaudeCodeCLI

                cli = ClaudeCodeCLI(cwd=temp_dir)

    seen = []

    async def mock_query(prompt, options):
        seen.append(options)
        yield {"subtype": "success", "result": "Hi"}

    with patch("src.claude_cli.query", mock_query):
        assert await cli.verify_cli() is True

    options = seen[0]
    assert options.tools == []
    assert options.setting_sources == []
    assert options.strict_mcp_config is True
    assert isinstance(options.system_prompt, str)

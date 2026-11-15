"""Session transcripts are written only when they can be resumed, and pruned.

The CLI writes a transcript per session so the session can be resumed, and its
own retention cleanup does not run for SDK sessions. The wrapper skips the
transcript where nothing resumes the session and deletes those the resume
index has forgotten.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src import main as main_mod
from src.claude_cli import _transcript_dir_name
from src.conversation_index import ConversationIndex

DAY = 86400


@pytest.fixture
def cli():
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("src.auth.validate_claude_code_auth") as mock_validate:
            with patch("src.auth.auth_manager") as mock_auth:
                mock_validate.return_value = (True, {"method": "anthropic"})
                mock_auth.get_claude_code_env_vars.return_value = {}
                from src.claude_cli import ClaudeCodeCLI

                yield ClaudeCodeCLI(cwd=temp_dir)


def _transcript(project: Path, name: str, age_seconds: float) -> Path:
    project.mkdir(parents=True, exist_ok=True)
    path = project / f"{name}.jsonl"
    path.write_text("{}\n")
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return path


class TestPruneTranscripts:
    @pytest.fixture
    def layout(self, cli, tmp_path, monkeypatch):
        temp_root = tmp_path / "tmp"
        temp_root.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
        projects = tmp_path / "config" / "projects"

        cli.cwd = temp_root / "claude_code_workspace_current"
        current = projects / _transcript_dir_name(str(cli.cwd))
        earlier_run = projects / _transcript_dir_name(
            str(temp_root / "claude_code_workspace_earlier")
        )
        user_project = projects / _transcript_dir_name(str(tmp_path / "work" / "app"))
        return cli, current, earlier_run, user_project

    def test_old_transcripts_of_the_current_workspace_are_deleted(self, layout):
        cli, current, _, _ = layout
        old = _transcript(current, "old", 2 * DAY)
        (current / "old" / "subagents").mkdir(parents=True)
        recent = _transcript(current, "recent", 60)

        assert cli.prune_transcripts(DAY) == 1

        assert not old.exists()
        assert not (current / "old").exists()
        assert recent.exists()

    def test_emptied_workspace_of_an_earlier_run_is_removed(self, layout):
        cli, current, earlier_run, _ = layout
        _transcript(earlier_run, "old", 2 * DAY)
        _transcript(current, "old", 2 * DAY)

        cli.prune_transcripts(DAY)

        assert not earlier_run.exists()
        assert current.exists()

    def test_transcripts_of_other_projects_are_never_touched(self, layout):
        cli, _, _, user_project = layout
        old = _transcript(user_project, "old", 30 * DAY)

        assert cli.prune_transcripts(DAY) == 0
        assert old.exists()

    def test_missing_projects_directory_is_fine(self, layout):
        cli, _, _, _ = layout
        assert cli.prune_transcripts(DAY) == 0


class TestSessionPersistence:
    @pytest.mark.asyncio
    async def test_run_completion_can_skip_the_transcript(self, cli):
        seen = []

        async def mock_query(prompt, options):
            seen.append(dict(options.extra_args))
            yield {"subtype": "success", "result": "ok"}

        with patch("src.claude_cli.query", mock_query):
            async for _ in cli.run_completion(prompt="hi"):
                pass
            async for _ in cli.run_completion(prompt="hi", persist_session=False):
                pass

        assert "no-session-persistence" not in seen[0]
        assert "no-session-persistence" in seen[1]

    @pytest.mark.asyncio
    async def test_auth_probe_writes_no_transcript(self, cli):
        seen = []

        async def mock_query(prompt, options):
            seen.append(options.extra_args)
            yield {"subtype": "success", "result": "Hi"}

        with patch("src.claude_cli.query", mock_query):
            await cli.verify_cli()

        assert "no-session-persistence" in seen[0]

    @pytest.mark.parametrize("enabled", [True, False])
    def test_chat_completions_keep_transcripts_only_when_resume_is_on(self, monkeypatch, enabled):
        monkeypatch.setattr(main_mod, "conversation_index", ConversationIndex(enabled=enabled))

        kwargs = main_mod._run_completion_kwargs({}, "hi", None, stream=False)

        assert kwargs["persist_session"] is enabled

    def test_anthropic_messages_writes_no_transcript(self, monkeypatch):
        monkeypatch.setattr(main_mod._auth, "cli_health", main_mod._auth.CliHealth())
        calls = []

        async def fake_run_completion(**kwargs):
            calls.append(kwargs)
            yield {"subtype": "success", "result": "Hi", "total_cost_usd": 0, "usage": None}

        monkeypatch.setattr(main_mod.claude_cli, "run_completion", fake_run_completion)
        TestClient(main_mod.app).post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert calls[0]["persist_session"] is False

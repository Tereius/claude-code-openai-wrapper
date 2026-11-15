"""Unit tests for model-list helpers and live default-model resolution."""

from types import SimpleNamespace

import pytest

from src import constants, main
from src.model_service import (
    ModelService,
    fallback_model_payload,
    openai_model_from_anthropic,
    pick_latest_sonnet,
)


@pytest.fixture
def fresh_service(monkeypatch):
    """Swap main's model_service for a fresh one with no credentials in the env."""
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_MODELS_OVERRIDE"):
        monkeypatch.delenv(name, raising=False)
    service = ModelService()
    monkeypatch.setattr(main, "model_service", service)
    monkeypatch.setattr(constants, "RESOLVED_DEFAULT_MODEL", None)
    monkeypatch.setattr(constants, "DEFAULT_MODEL_ENV", None)
    return service


def _set_auth(monkeypatch, method):
    fake = SimpleNamespace(auth_method=method)
    monkeypatch.setattr("src.model_service.auth_manager", fake)
    monkeypatch.setattr(main._auth, "auth_manager", fake)


SONNETS = [
    {
        "id": "claude-sonnet-4-7",
        "object": "model",
        "owned_by": "anthropic",
        "created_at": "2026-06-01T00:00:00Z",
    },
    {
        "id": "claude-sonnet-4-6",
        "object": "model",
        "owned_by": "anthropic",
        "created_at": "2026-04-01T00:00:00Z",
    },
]


def test_openai_model_from_anthropic_preserves_metadata():
    model = openai_model_from_anthropic(
        {
            "id": "claude-test",
            "type": "model",
            "display_name": "Claude Test",
            "created_at": "2026-01-01T00:00:00Z",
            "max_input_tokens": 200000,
            "max_tokens": 64000,
            "capabilities": {"batch": {"supported": True}},
        }
    )

    assert model["id"] == "claude-test"
    assert model["object"] == "model"
    assert model["owned_by"] == "anthropic"
    # `created` should be the unix timestamp of the ISO `created_at`.
    assert model["created"] == 1767225600
    assert model["capabilities"] == {"batch": {"supported": True}}


def test_fallback_objects_include_created_field():
    fallback = fallback_model_payload()

    assert fallback, "fallback list should not be empty"
    for entry in fallback:
        assert isinstance(entry["created"], int) and entry["created"] > 0


def test_pick_latest_sonnet_prefers_newest_created_at():
    models = [
        {"id": "claude-sonnet-4-5", "created_at": "2025-09-29T00:00:00Z"},
        {"id": "claude-sonnet-4-6", "created_at": "2026-04-01T00:00:00Z"},
        {"id": "claude-opus-4-6", "created_at": "2026-04-15T00:00:00Z"},
    ]

    assert pick_latest_sonnet(models) == "claude-sonnet-4-6"


def test_pick_latest_sonnet_returns_none_when_no_sonnet():
    models = [{"id": "claude-haiku-4-5", "created_at": "2025-10-01T00:00:00Z"}]

    assert pick_latest_sonnet(models) is None


@pytest.mark.asyncio
async def test_list_models_endpoint_uses_model_service(fresh_service, monkeypatch):
    async def fake_available():
        return [{"id": "claude-live", "object": "model", "owned_by": "anthropic"}]

    async def no_auth(request, credentials):
        return True

    monkeypatch.setattr(fresh_service, "get_available_models", fake_available)
    monkeypatch.setattr(main, "verify_api_key", no_auth)

    result = await main.list_models(None, None)

    assert result == {"object": "list", "data": await fake_available()}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth_method,env_var,value",
    [
        ("anthropic", "ANTHROPIC_API_KEY", "sk-ant-api-test"),
        ("claude_cli", "CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test"),
    ],
)
async def test_resolve_default_model_sets_constants(
    fresh_service, monkeypatch, auth_method, env_var, value
):
    async def fake_fetch():
        return SONNETS

    _set_auth(monkeypatch, auth_method)
    monkeypatch.setenv(env_var, value)
    monkeypatch.setattr(fresh_service, "fetch_models_from_api", fake_fetch)

    resolved = await main.resolve_default_model()

    assert resolved == "claude-sonnet-4-7"
    assert constants.RESOLVED_DEFAULT_MODEL == "claude-sonnet-4-7"


@pytest.mark.asyncio
async def test_resolve_default_model_skips_without_credentials(fresh_service, monkeypatch, caplog):
    """No API key or OAuth token -> skip live discovery, log clearly, use fallback."""

    async def fake_fetch():
        raise AssertionError("should not call live API without credentials")

    _set_auth(monkeypatch, "claude_cli")
    monkeypatch.setattr(fresh_service, "fetch_models_from_api", fake_fetch)

    with caplog.at_level("INFO", logger="src.main"):
        resolved = await main.resolve_default_model()

    assert resolved is None
    assert constants.RESOLVED_DEFAULT_MODEL is None
    assert any("Live model discovery disabled" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_resolve_default_model_skips_bedrock_even_with_token(fresh_service, monkeypatch):
    async def fake_fetch():
        raise AssertionError("bedrock must not use the Anthropic catalogue")

    _set_auth(monkeypatch, "bedrock")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
    monkeypatch.setattr(fresh_service, "fetch_models_from_api", fake_fetch)

    assert await main.resolve_default_model() is None


@pytest.mark.asyncio
async def test_resolve_default_model_honors_env_override(fresh_service, monkeypatch):
    async def fake_fetch():
        raise AssertionError("env override should short-circuit fetch")

    monkeypatch.setattr(constants, "DEFAULT_MODEL_ENV", "claude-opus-4-6")
    monkeypatch.setattr(fresh_service, "fetch_models_from_api", fake_fetch)

    resolved = await main.resolve_default_model()

    assert resolved == "claude-opus-4-6"
    assert constants.RESOLVED_DEFAULT_MODEL is None


def test_get_default_model_prefers_resolved_over_fallback(monkeypatch):
    from src import models as models_module

    monkeypatch.setattr(constants, "DEFAULT_MODEL_ENV", None)
    monkeypatch.setattr(constants, "RESOLVED_DEFAULT_MODEL", "claude-sonnet-future")

    assert models_module.get_default_model() == "claude-sonnet-future"


def test_get_default_model_env_override_wins(monkeypatch):
    from src import models as models_module

    monkeypatch.setattr(constants, "DEFAULT_MODEL_ENV", "claude-opus-4-6")
    monkeypatch.setattr(constants, "RESOLVED_DEFAULT_MODEL", "claude-sonnet-future")

    assert models_module.get_default_model() == "claude-opus-4-6"

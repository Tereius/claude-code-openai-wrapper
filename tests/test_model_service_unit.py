#!/usr/bin/env python3
"""
Unit tests for src/model_service.py

ModelService owns live model discovery: credential selection per auth method
(API key or claude.ai OAuth token), the paginated Models API fetch, the TTL
cache with stale-on-error behaviour, and the refresh/status endpoints.
HTTP is served by httpx.MockTransport, so nothing touches the network.
"""

import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest

from src import constants
from src.model_service import OAUTH_BETA_HEADER, ModelService

MODELS_URL = "https://api.anthropic.com/v1/models"


def _model(model_id, created_at="2026-01-01T00:00:00Z"):
    return {"type": "model", "id": model_id, "display_name": model_id, "created_at": created_at}


def _page(*ids, has_more=False):
    return {
        "data": [_model(i) for i in ids],
        "has_more": has_more,
        "first_id": ids[0] if ids else None,
        "last_id": ids[-1] if ids else None,
    }


@pytest.fixture
def env(monkeypatch):
    """Start every test with no discovery credentials and no override.

    main.py calls load_dotenv(), so a developer's real .env could otherwise
    leak ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN into the test process.
    """
    for name in (
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_MODELS_OVERRIDE",
        "ANTHROPIC_BETA",
        "ANTHROPIC_BETA_HEADER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(constants, "ANTHROPIC_MODELS_URL", MODELS_URL)
    monkeypatch.setattr(constants, "MODEL_LIST_CACHE_TTL_SECONDS", 3600)
    monkeypatch.setattr(constants, "MODEL_LIST_ERROR_TTL_SECONDS", 60)
    return monkeypatch


def _set_auth(monkeypatch, method):
    monkeypatch.setattr("src.model_service.auth_manager", SimpleNamespace(auth_method=method))


def _service_with(handler):
    """A ModelService whose HTTP client is served by `handler(request)`."""
    service = ModelService()
    service._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return service


def _recording_handler(responses):
    """Serve `responses` in order and record each request."""
    requests = []
    queue = list(responses)

    def handler(request):
        requests.append(request)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        status, body = item
        return httpx.Response(status, json=body)

    return handler, requests


class TestDiscoveryCredentials:
    def test_api_key_auth_uses_x_api_key(self, env):
        _set_auth(env, "anthropic")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")

        name, headers = ModelService().discovery_credentials()

        assert name == "api_key"
        assert headers == {"x-api-key": "sk-ant-api-test"}

    def test_cli_auth_uses_oauth_token_as_bearer(self, env):
        _set_auth(env, "claude_cli")
        env.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")

        name, headers = ModelService().discovery_credentials()

        assert name == "oauth_token"
        assert headers["authorization"] == "Bearer sk-ant-oat01-test"
        assert headers["anthropic-beta"] == OAUTH_BETA_HEADER
        assert "x-api-key" not in headers

    def test_cli_auth_prefers_oauth_token_over_api_key(self, env):
        _set_auth(env, "claude_cli")
        env.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")

        name, _ = ModelService().discovery_credentials()

        assert name == "oauth_token"

    def test_cli_auth_falls_back_to_api_key(self, env):
        _set_auth(env, "claude_cli")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")

        name, _ = ModelService().discovery_credentials()

        assert name == "api_key"

    def test_api_key_auth_ignores_oauth_token(self, env):
        _set_auth(env, "anthropic")
        env.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")

        assert ModelService().discovery_credentials() is None

    def test_cli_auth_without_credentials(self, env):
        _set_auth(env, "claude_cli")

        assert ModelService().discovery_credentials() is None

    @pytest.mark.parametrize("method", ["bedrock", "vertex", "unknown_method"])
    def test_other_auth_methods_never_discover(self, env, method):
        _set_auth(env, method)
        env.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")

        assert ModelService().discovery_credentials() is None


class TestFetchModels:
    @pytest.mark.asyncio
    async def test_oauth_request_headers(self, env):
        _set_auth(env, "claude_cli")
        env.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
        handler, requests = _recording_handler([(200, _page("claude-a"))])
        service = _service_with(handler)

        result = await service.fetch_models_from_api()

        assert [m["id"] for m in result] == ["claude-a"]
        req = requests[0]
        assert str(req.url).startswith(MODELS_URL)
        assert req.headers["authorization"] == "Bearer sk-ant-oat01-test"
        assert req.headers["anthropic-beta"] == OAUTH_BETA_HEADER
        assert req.headers["anthropic-version"] == constants.ANTHROPIC_VERSION
        assert "x-api-key" not in req.headers
        assert service._credential == "oauth_token"

    @pytest.mark.asyncio
    async def test_extra_beta_header_is_merged(self, env):
        _set_auth(env, "claude_cli")
        env.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
        env.setenv("ANTHROPIC_BETA", "some-beta-2026")
        handler, requests = _recording_handler([(200, _page("claude-a"))])

        await _service_with(handler).fetch_models_from_api()

        assert requests[0].headers["anthropic-beta"] == f"{OAUTH_BETA_HEADER},some-beta-2026"

    @pytest.mark.asyncio
    async def test_api_key_request_headers(self, env):
        _set_auth(env, "anthropic")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
        handler, requests = _recording_handler([(200, _page("claude-a"))])

        await _service_with(handler).fetch_models_from_api()

        assert requests[0].headers["x-api-key"] == "sk-ant-api-test"
        assert "authorization" not in requests[0].headers

    @pytest.mark.asyncio
    async def test_follows_pagination(self, env):
        _set_auth(env, "anthropic")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
        handler, requests = _recording_handler(
            [(200, _page("claude-a", "claude-b", has_more=True)), (200, _page("claude-c"))]
        )

        result = await _service_with(handler).fetch_models_from_api()

        assert [m["id"] for m in result] == ["claude-a", "claude-b", "claude-c"]
        assert len(requests) == 2
        assert requests[1].url.params["after_id"] == "claude-b"

    @pytest.mark.asyncio
    async def test_preserves_metadata(self, env):
        _set_auth(env, "anthropic")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
        body = {
            "data": [
                {
                    **_model("claude-a"),
                    "max_input_tokens": 1_000_000,
                    "capabilities": {"batch": {"supported": True}},
                }
            ],
            "has_more": False,
        }
        handler, _ = _recording_handler([(200, body)])

        [model] = await _service_with(handler).fetch_models_from_api()

        assert model["object"] == "model"
        assert model["max_input_tokens"] == 1_000_000
        assert model["capabilities"] == {"batch": {"supported": True}}

    @pytest.mark.asyncio
    async def test_no_credentials_makes_no_request(self, env):
        _set_auth(env, "claude_cli")
        handler, requests = _recording_handler([(200, _page("claude-a"))])

        assert await _service_with(handler).fetch_models_from_api() is None
        assert requests == []

    @pytest.mark.asyncio
    async def test_oauth_401_names_the_token(self, env, caplog):
        _set_auth(env, "claude_cli")
        env.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-expired")
        handler, _ = _recording_handler([(401, {"type": "error"})])

        with caplog.at_level("WARNING", logger="src.model_service"):
            result = await _service_with(handler).fetch_models_from_api()

        assert result is None
        assert any("CLAUDE_CODE_OAUTH_TOKEN" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        [
            (429, {"type": "error"}),
            (500, {"type": "error"}),
            (200, {"data": []}),
            httpx.TimeoutException("timeout"),
            httpx.ConnectError("connection failed"),
        ],
    )
    async def test_failures_return_none(self, env, response):
        _set_auth(env, "anthropic")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
        handler, _ = _recording_handler([response])

        assert await _service_with(handler).fetch_models_from_api() is None


class TestCache:
    @pytest.mark.asyncio
    async def test_success_is_cached_for_full_ttl(self, env):
        _set_auth(env, "claude_cli")
        env.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
        handler, requests = _recording_handler([(200, _page("claude-a"))])
        service = _service_with(handler)

        first = await service.get_available_models()
        second = await service.get_available_models()

        assert [m["id"] for m in first] == [m["id"] for m in second] == ["claude-a"]
        assert len(requests) == 1
        assert service._expires_at - time.time() > 3000
        assert service._source == "api"

    @pytest.mark.asyncio
    async def test_failure_without_live_list_serves_static_with_short_ttl(self, env):
        _set_auth(env, "anthropic")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
        handler, _ = _recording_handler([(500, {})])
        service = _service_with(handler)

        models = await service.get_available_models()

        assert [m["id"] for m in models] == list(constants.CLAUDE_MODELS)
        assert service._expires_at - time.time() < 120
        assert service._source == "fallback"

    @pytest.mark.asyncio
    async def test_failure_keeps_stale_live_list(self, env):
        _set_auth(env, "anthropic")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
        handler, _ = _recording_handler([(200, _page("claude-a")), (500, {})])
        service = _service_with(handler)

        await service.get_available_models()
        service._expires_at = 0  # force expiry
        models = await service.get_available_models()

        assert [m["id"] for m in models] == ["claude-a"]
        assert service._source == "api"
        assert service._expires_at - time.time() < 120

    @pytest.mark.asyncio
    async def test_concurrent_calls_only_fetch_once(self, env):
        _set_auth(env, "anthropic")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
        service = ModelService()
        calls = 0

        async def fake_fetch():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return [{"id": "claude-a", "object": "model", "owned_by": "anthropic"}]

        service.fetch_models_from_api = fake_fetch

        results = await asyncio.gather(*[service.get_available_models() for _ in range(8)])

        assert calls == 1
        assert all(r[0]["id"] == "claude-a" for r in results)

    @pytest.mark.asyncio
    async def test_override_skips_live_fetch(self, env):
        _set_auth(env, "claude_cli")
        env.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
        env.setenv("CLAUDE_MODELS_OVERRIDE", "custom-a,custom-b")
        env.setattr(constants, "CLAUDE_MODELS", ["custom-a", "custom-b"])
        handler, requests = _recording_handler([(200, _page("claude-a"))])
        service = _service_with(handler)

        models = await service.get_available_models()

        assert [m["id"] for m in models] == ["custom-a", "custom-b"]
        assert service.get_models() == ["custom-a", "custom-b"]
        assert service.get_status()["source"] == "override"
        assert requests == []

    @pytest.mark.asyncio
    async def test_get_models_returns_live_ids(self, env):
        _set_auth(env, "claude_cli")
        env.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
        handler, _ = _recording_handler([(200, _page("claude-a", "claude-b"))])
        service = _service_with(handler)

        assert service.get_models() == list(constants.CLAUDE_MODELS)
        await service.get_available_models()
        assert service.get_models() == ["claude-a", "claude-b"]


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_populates_cache_and_status(self, env):
        _set_auth(env, "claude_cli")
        env.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
        handler, _ = _recording_handler([(200, _page("claude-a"))])
        service = _service_with(handler)

        await service.initialize()
        status = service.get_status()

        assert service.is_initialized()
        assert status["source"] == "api"
        assert status["credential"] == "oauth_token"
        assert status["auth_method"] == "claude_cli"
        assert status["model_count"] == 1
        assert status["last_refresh"] is not None

    @pytest.mark.asyncio
    async def test_initialize_only_once(self, env):
        _set_auth(env, "anthropic")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
        handler, requests = _recording_handler([(200, _page("claude-a"))])
        service = _service_with(handler)

        await service.initialize()
        service._expires_at = 0
        await service.initialize()

        assert len(requests) == 1

    @pytest.mark.asyncio
    async def test_initialize_without_credentials_uses_fallback(self, env):
        _set_auth(env, "claude_cli")
        service = ModelService()

        await service.initialize()
        status = service.get_status()

        assert status["source"] == "fallback"
        assert status["credential"] is None
        assert service.get_models() == list(constants.CLAUDE_MODELS)
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_resets_state(self, env):
        _set_auth(env, "anthropic")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
        handler, _ = _recording_handler([(200, _page("claude-a"))])
        service = _service_with(handler)
        await service.initialize()

        await service.shutdown()

        assert not service.is_initialized()
        assert service._http_client is None
        assert service.get_models() == list(constants.CLAUDE_MODELS)
        assert service.get_status()["source"] == "fallback"
        assert service.get_status()["last_refresh"] is None

    @pytest.mark.asyncio
    async def test_shutdown_safe_when_not_initialized(self):
        service = ModelService()
        await service.shutdown()
        assert service._http_client is None


class TestRefresh:
    @pytest.mark.asyncio
    async def test_refresh_ignores_fresh_cache(self, env):
        _set_auth(env, "claude_cli")
        env.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
        handler, requests = _recording_handler(
            [(200, _page("claude-a")), (200, _page("claude-a", "claude-new"))]
        )
        service = _service_with(handler)
        await service.get_available_models()

        result = await service.refresh_models()

        assert len(requests) == 2
        assert result["success"] is True
        assert result["models"] == ["claude-a", "claude-new"]
        assert result["credential"] == "oauth_token"
        assert result["auth_method"] == "claude_cli"

    @pytest.mark.asyncio
    async def test_refresh_failure_preserves_models(self, env):
        _set_auth(env, "anthropic")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
        handler, _ = _recording_handler([(200, _page("claude-a", "claude-b")), (500, {})])
        service = _service_with(handler)
        await service.get_available_models()
        last_refresh = service._last_refresh

        result = await service.refresh_models()

        assert result["success"] is False
        assert "API fetch failed" in result["message"]
        assert result["current_count"] == 2
        assert result["source"] == "api"
        assert service.get_models() == ["claude-a", "claude-b"]
        assert service._last_refresh == last_refresh

    @pytest.mark.asyncio
    async def test_refresh_cli_without_credentials(self, env):
        _set_auth(env, "claude_cli")

        result = await ModelService().refresh_models()

        assert result["success"] is False
        assert "CLAUDE_CODE_OAUTH_TOKEN" in result["message"]
        assert result["auth_method"] == "claude_cli"
        assert result["current_count"] == len(constants.CLAUDE_MODELS)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["bedrock", "vertex"])
    async def test_refresh_unsupported_auth(self, env, method):
        _set_auth(env, method)

        result = await ModelService().refresh_models()

        assert result["success"] is False
        assert f"not supported for auth method: {method}" in result["message"]
        assert result["auth_method"] == method

    @pytest.mark.asyncio
    async def test_refresh_with_override(self, env):
        _set_auth(env, "anthropic")
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
        env.setenv("CLAUDE_MODELS_OVERRIDE", "custom-a")

        result = await ModelService().refresh_models()

        assert result["success"] is False
        assert result["source"] == "override"

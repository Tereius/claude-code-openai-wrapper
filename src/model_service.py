"""
Model discovery: the single source of truth for which models the wrapper offers.

Fetches Anthropic's live Models API (``GET /v1/models``) and falls back to the
static ``CLAUDE_MODELS`` list from constants when live discovery is unavailable.

Credentials used for discovery, by auth method:

- ``anthropic``:  ``ANTHROPIC_API_KEY`` sent as ``x-api-key``.
- ``claude_cli``: ``CLAUDE_CODE_OAUTH_TOKEN`` (a claude.ai subscription token)
  sent as ``Authorization: Bearer`` plus the OAuth beta header, or
  ``ANTHROPIC_API_KEY`` when no token is set.
- ``bedrock`` / ``vertex``: none. Those platforms use different model IDs, so
  the Anthropic catalogue would advertise names they can't serve.

Results are cached for ``MODEL_LIST_CACHE_TTL_SECONDS``. A failed fetch keeps
serving the last good live list (or the static list if there never was one)
and retries after ``MODEL_LIST_ERROR_TTL_SECONDS``.
"""

import asyncio
import os
import time
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple

import httpx

from src import constants
from src.auth import auth_manager

logger = logging.getLogger(__name__)

# Beta flag Claude Code sends when it calls the API with a subscription token.
OAUTH_BETA_HEADER = "oauth-2025-04-20"

# Auth methods whose models are named like the Anthropic catalogue.
DISCOVERY_AUTH_METHODS = ("anthropic", "claude_cli")


def _iso_to_unix(value: Any) -> Optional[int]:
    """Convert an Anthropic ISO-8601 'created_at' string to a unix timestamp."""
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def openai_model_from_anthropic(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic ModelInfo object to OpenAI-compatible model metadata."""
    created = _iso_to_unix(model_info.get("created_at"))
    model: Dict[str, Any] = {
        "id": model_info["id"],
        "object": "model",
        "created": created if created is not None else int(datetime.now(timezone.utc).timestamp()),
        "owned_by": "anthropic",
    }

    # Preserve useful Anthropic metadata for clients that want it.  OpenAI clients
    # ignore unknown keys, and the existing id/object/owned_by shape is retained.
    for key in (
        "display_name",
        "created_at",
        "max_input_tokens",
        "max_tokens",
        "capabilities",
        "type",
    ):
        if key in model_info:
            model[key] = model_info[key]

    return model


def fallback_model_payload() -> List[Dict[str, Any]]:
    """The static model list in OpenAI /v1/models shape."""
    now = int(datetime.now(timezone.utc).timestamp())
    return [
        {"id": model_id, "object": "model", "created": now, "owned_by": "anthropic"}
        for model_id in constants.CLAUDE_MODELS
    ]


def pick_latest_sonnet(models: List[Dict[str, Any]]) -> Optional[str]:
    """Return the id of the newest Sonnet model in `models`, or None."""
    sonnets = [m for m in models if isinstance(m.get("id"), str) and "sonnet" in m["id"].lower()]
    if not sonnets:
        return None
    # Prefer Anthropic-provided created_at; fall back to the int `created` we set,
    # then to id-sort (date-suffixed ids sort correctly newest-last).
    sonnets.sort(
        key=lambda m: (
            _iso_to_unix(m.get("created_at")) or m.get("created") or 0,
            m["id"],
        )
    )
    return sonnets[-1]["id"]


def _models_override_active() -> bool:
    return bool(os.getenv("CLAUDE_MODELS_OVERRIDE", "").strip())


class ModelService:
    """Fetches models from the Anthropic Models API with fallback to constants."""

    def __init__(self) -> None:
        # Last successful live fetch (OpenAI-shaped dicts), or None.
        self._live_models: Optional[List[Dict[str, Any]]] = None
        self._expires_at: float = 0.0
        self._lock: Optional[asyncio.Lock] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._initialized: bool = False
        self._last_refresh: Optional[float] = None
        self._source: str = "fallback"  # "api", "fallback" or "override"
        self._credential: Optional[str] = None  # credential used by the last live fetch

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Called during app startup: populate the cache from the live API if possible."""
        if self._initialized:
            return

        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=constants.MODEL_LIST_REQUEST_TIMEOUT_SECONDS
            )
        await self.get_available_models()

        if self._source == "api":
            logger.info(
                "Fetched %d models from Anthropic Models API (credential: %s)",
                len(self._live_models or []),
                self._credential,
            )
        elif self._source == "override":
            logger.info("Using CLAUDE_MODELS_OVERRIDE model list")
        else:
            logger.info("Using fallback static model list from constants")

        self._initialized = True

    async def shutdown(self) -> None:
        """Close HTTP client and drop cached state on app shutdown."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._live_models = None
        self._expires_at = 0.0
        self._initialized = False
        self._last_refresh = None
        self._source = "fallback"
        self._credential = None

    # ------------------------------------------------------------------
    # Credentials and fetching
    # ------------------------------------------------------------------

    def discovery_credentials(self) -> Optional[Tuple[str, Dict[str, str]]]:
        """Pick the credential for the Models API.

        Returns (credential_name, auth_headers), or None when live discovery
        isn't possible for the configured auth method.
        """
        auth_method = auth_manager.auth_method
        if auth_method not in DISCOVERY_AUTH_METHODS:
            return None

        api_key = os.getenv("ANTHROPIC_API_KEY")
        oauth_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")

        # In CLI mode the subscription token decides which models the user can
        # run, so prefer it; in API-key mode the key is what the CLI uses.
        if auth_method == "claude_cli" and oauth_token:
            return "oauth_token", {
                "authorization": f"Bearer {oauth_token}",
                "anthropic-beta": OAUTH_BETA_HEADER,
            }
        if api_key:
            return "api_key", {"x-api-key": api_key}
        return None

    def _request_headers(self, auth_headers: Dict[str, str]) -> Dict[str, str]:
        headers = {"anthropic-version": constants.ANTHROPIC_VERSION, **auth_headers}
        extra_beta = os.getenv("ANTHROPIC_BETA") or os.getenv("ANTHROPIC_BETA_HEADER")
        if extra_beta:
            existing = headers.get("anthropic-beta")
            headers["anthropic-beta"] = f"{existing},{extra_beta}" if existing else extra_beta
        return headers

    async def fetch_models_from_api(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch the full live catalogue, following pagination.

        Returns OpenAI-shaped model dicts on success, None when discovery is
        unavailable or the request fails (callers fall back gracefully).
        """
        creds = self.discovery_credentials()
        if creds is None:
            logger.debug(
                "No Models API credential for auth method '%s'; using static model list",
                auth_manager.auth_method,
            )
            return None
        credential, auth_headers = creds
        headers = self._request_headers(auth_headers)

        if not self._http_client:
            self._http_client = httpx.AsyncClient(
                timeout=constants.MODEL_LIST_REQUEST_TIMEOUT_SECONDS
            )

        params: Dict[str, Any] = {"limit": 1000}
        models: List[Dict[str, Any]] = []
        try:
            while True:
                response = await self._http_client.get(
                    constants.ANTHROPIC_MODELS_URL, headers=headers, params=params
                )
                if response.status_code != 200:
                    hint = ""
                    if response.status_code in (401, 403):
                        env = (
                            "CLAUDE_CODE_OAUTH_TOKEN"
                            if credential == "oauth_token"
                            else "ANTHROPIC_API_KEY"
                        )
                        hint = f" Check {env}."
                    logger.warning(
                        "Anthropic Models API returned %s.%s Using fallback models.",
                        response.status_code,
                        hint,
                    )
                    return None

                payload = response.json()
                models.extend(
                    openai_model_from_anthropic(model)
                    for model in payload.get("data", [])
                    if model.get("id")
                )
                if not payload.get("has_more") or not payload.get("last_id"):
                    break
                params["after_id"] = payload["last_id"]
        except httpx.TimeoutException:
            logger.warning(
                "Anthropic Models API timed out after %ss",
                constants.MODEL_LIST_REQUEST_TIMEOUT_SECONDS,
            )
            return None
        except Exception as exc:  # noqa: BLE001 - discovery must degrade gracefully
            logger.warning("Failed to fetch Anthropic model list, using fallback: %s", exc)
            return None

        if not models:
            logger.warning("Anthropic Models API returned an empty model list")
            return None

        self._credential = credential
        return models

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _update_from_api(self) -> bool:
        """Fetch and update the cache. Caller must hold the lock."""
        now = time.time()
        models = await self.fetch_models_from_api()
        if models:
            self._live_models = models
            self._expires_at = now + constants.MODEL_LIST_CACHE_TTL_SECONDS
            self._last_refresh = now
            self._source = "api"
            return True

        # Keep whatever we had (stale live list or static fallback) and retry
        # soon, so a transient blip doesn't suppress live discovery for an hour.
        self._expires_at = now + constants.MODEL_LIST_ERROR_TTL_SECONDS
        if self._live_models is None:
            self._source = "fallback"
        return False

    def _current_payload(self) -> List[Dict[str, Any]]:
        return self._live_models if self._live_models else fallback_model_payload()

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Models for GET /v1/models: live when possible, cached, static fallback."""
        if _models_override_active():
            self._source = "override"
            return fallback_model_payload()

        if time.time() < self._expires_at:
            return self._current_payload()

        async with self._get_lock():
            # Recheck inside the lock so the first waiter populates the cache and
            # subsequent waiters return without re-fetching.
            if time.time() >= self._expires_at:
                await self._update_from_api()
            return self._current_payload()

    def get_models(self) -> List[str]:
        """Model IDs for request validation (sync; uses whatever is cached)."""
        if self._live_models and not _models_override_active():
            return [m["id"] for m in self._live_models]
        return list(constants.CLAUDE_MODELS)

    def is_initialized(self) -> bool:
        """Check if service has been initialized."""
        return self._initialized

    async def refresh_models(self) -> Dict[str, Any]:
        """Force a live refresh, ignoring the cache TTL.

        If the fetch fails, the existing models are kept.
        """
        auth_method = auth_manager.auth_method

        if _models_override_active():
            return {
                "success": False,
                "message": "CLAUDE_MODELS_OVERRIDE is set; live discovery is disabled",
                "current_count": len(self.get_models()),
                "source": "override",
                "auth_method": auth_method,
            }

        if self.discovery_credentials() is None:
            if auth_method in DISCOVERY_AUTH_METHODS:
                message = (
                    "Dynamic refresh requires ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN. "
                    f"Current auth: {auth_method}"
                )
            else:
                message = f"Dynamic refresh is not supported for auth method: {auth_method}"
            return {
                "success": False,
                "message": message,
                "current_count": len(self.get_models()),
                "source": self._source,
                "auth_method": auth_method,
            }

        async with self._get_lock():
            ok = await self._update_from_api()

        if ok:
            models = self.get_models()
            logger.info("Refreshed %d models from Anthropic Models API", len(models))
            return {
                "success": True,
                "count": len(models),
                "source": "api",
                "models": models,
                "auth_method": auth_method,
                "credential": self._credential,
            }
        return {
            "success": False,
            "message": "API fetch failed, keeping existing models",
            "current_count": len(self.get_models()),
            "source": self._source,
            "auth_method": auth_method,
        }

    def get_status(self) -> Dict[str, Any]:
        """Service status: source, auth method, credential and last refresh time."""
        return {
            "initialized": self._initialized,
            "source": "override" if _models_override_active() else self._source,
            "model_count": len(self.get_models()),
            "last_refresh": self._last_refresh,
            "auth_method": auth_manager.auth_method,
            "credential": self._credential if self._source == "api" else None,
        }


# Global singleton instance
model_service = ModelService()

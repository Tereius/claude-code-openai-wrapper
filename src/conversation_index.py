"""Find the CLI session that already holds a conversation's history.

OpenAI clients resend the whole conversation on every turn. Flattened into one
prompt string, that history sits inside a single user message that grows each
turn, so the prompt cache can never match it and every turn pays to write the
full history to the cache again. Resuming the CLI session that answered the
previous turn sends only the new message; the history then comes from the
cache, as it does in Claude Code itself.

After each successful completion the index maps the conversation as the
client will send it back (request messages plus the reply it received) to the
CLI session that produced the reply. A later request whose messages up to its
last assistant message match an entry resumes that session. Sessions are
resumed with fork_session, so a regenerated or edited turn branches off
without touching the session other branches build on.
"""

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, List, Optional

from src.models import Message

logger = logging.getLogger(__name__)


@dataclass
class ResumePlan:
    """The CLI session to resume and the messages it has not seen yet."""

    session_id: str
    new_messages: List[Message]


def _tool_call_key(tool_call: Any) -> List[Any]:
    function = tool_call.function if hasattr(tool_call, "function") else tool_call["function"]
    name = function.name if hasattr(function, "name") else function.get("name")
    arguments = function.arguments if hasattr(function, "arguments") else function.get("arguments")
    # Clients may re-serialise the arguments; compare them as JSON values.
    try:
        arguments = json.loads(arguments)
    except (TypeError, ValueError):
        pass
    # Call ids are left out: they are generated per reply and add nothing.
    return [name, arguments]


def _message_key(message: Message) -> List[Any]:
    # Clients commonly trim or reflow whitespace of the reply they display.
    content = " ".join(message.content.split()) if isinstance(message.content, str) else ""
    tool_calls = [_tool_call_key(tc) for tc in message.tool_calls or []]
    tool_call_id = message.tool_call_id if message.role == "tool" else None
    return [message.role, content, tool_calls, tool_call_id]


def conversation_key(messages: List[Message]) -> str:
    payload = json.dumps([_message_key(m) for m in messages], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def describe(messages: List[Message]) -> str:
    """Role, short fingerprint and length of each message, for the log.

    Comparing the line logged when a reply is recorded with the one logged
    when the next turn finds no session shows which message the client
    changed in between.
    """
    parts = []
    for m in messages:
        digest = hashlib.sha256(
            json.dumps(_message_key(m), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:6]
        length = len(m.content) if isinstance(m.content, str) else 0
        parts.append(f"{m.role}:{digest}/{length}")
    return " ".join(parts)


class ConversationIndex:
    """Bounded in-memory map from conversation fingerprint to CLI session id."""

    def __init__(self, enabled: bool = True, max_entries: int = 1000, ttl_seconds: float = 86400):
        self.enabled = enabled
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
        self._lock = Lock()

    def lookup(self, messages: List[Message]) -> Optional[ResumePlan]:
        """Plan a resume when only user or tool messages follow the last reply."""
        if not self.enabled:
            return None
        last_assistant = max(
            (i for i, m in enumerate(messages) if m.role == "assistant"), default=None
        )
        if last_assistant is None:
            return None
        new_messages = messages[last_assistant + 1 :]
        if not new_messages or any(m.role not in ("user", "tool") for m in new_messages):
            return None

        history = messages[: last_assistant + 1]
        key = conversation_key(history)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and time.monotonic() - entry[1] > self.ttl_seconds:
                del self._entries[key]
                entry = None
            if entry is not None:
                self._entries.move_to_end(key)
        if entry is None:
            logger.info(f"No CLI session to resume for [{describe(history)}]")
            return None
        return ResumePlan(session_id=entry[0], new_messages=new_messages)

    def record(self, messages: List[Message], reply: Message, session_id: Optional[str]) -> None:
        """Remember the session that produced ``reply`` to ``messages``."""
        if not self.enabled or not session_id:
            return
        if not reply.content and not reply.tool_calls:
            return
        conversation = list(messages) + [reply]
        key = conversation_key(conversation)
        logger.info(f"Recorded CLI session {session_id} for [{describe(conversation)}]")
        with self._lock:
            self._entries[key] = (session_id, time.monotonic())
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


conversation_index = ConversationIndex(
    enabled=os.getenv("CONVERSATION_RESUME", "true").lower() not in ("false", "0", "no"),
    max_entries=int(os.getenv("CONVERSATION_RESUME_MAX_ENTRIES", "1000")),
    ttl_seconds=float(os.getenv("CONVERSATION_RESUME_TTL_HOURS", "24")) * 3600,
)

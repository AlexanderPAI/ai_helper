"""Per-chat conversation sessions and concurrency control."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from uuid import UUID, uuid4


class ChatSessionRegistry:
    """Keep one agent session and one processing lock for each active chat."""

    def __init__(self) -> None:
        self._session_ids: dict[int, UUID] = {}
        self._chat_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    def session_id(self, chat_id: int) -> UUID:
        """Return the current session, creating it on first use."""
        return self._session_ids.setdefault(chat_id, uuid4())

    def reset(self, chat_id: int) -> UUID:
        """Replace and return the current session for a chat."""
        session_id = uuid4()
        self._session_ids[chat_id] = session_id
        return session_id

    def lock(self, chat_id: int) -> asyncio.Lock:
        """Serialize agent requests that share conversation history."""
        return self._chat_locks[chat_id]


__all__ = ["ChatSessionRegistry"]

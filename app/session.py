from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional


@dataclass
class ConversationState:
    active_province: Optional[str] = None
    history: List[str] = field(default_factory=list)


class SessionStore:
    def __init__(self, max_turns: int = 6) -> None:
        self._store: Dict[str, ConversationState] = {}
        self._max_turns = max_turns
        self._lock = Lock()

    def get(self, session_id: str) -> ConversationState:
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = ConversationState()
            return self._store[session_id]

    def append_turn(self, session_id: str, user_query: str, bot_reply: str) -> None:
        state = self.get(session_id)
        pair = f"Q: {user_query}\nA: {bot_reply}"
        with self._lock:
            state.history.append(pair)
            if len(state.history) > self._max_turns:
                state.history = state.history[-self._max_turns :]


from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models.conversation_state import ConversationState


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db

    def get(self, session_id: str) -> ConversationState | None:
        return self.db.scalar(select(ConversationState).where(ConversationState.session_id == session_id))

    def upsert(self, state: ConversationState) -> ConversationState:
        self.db.add(state)
        self.db.flush()
        return state

    def clear_expired(self, ttl_minutes: int) -> None:
        cutoff = datetime.utcnow() - timedelta(minutes=ttl_minutes)
        self.db.execute(delete(ConversationState).where(ConversationState.updated_at < cutoff))

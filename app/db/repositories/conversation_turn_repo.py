from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.models.conversation_turn import ConversationTurn


class ConversationTurnRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_turns(self, session_id: str, limit: int = 10) -> list[ConversationTurn]:
        return list(
            self.db.scalars(
                select(ConversationTurn)
                .where(ConversationTurn.session_id == session_id)
                .order_by(ConversationTurn.turn_index.desc())
                .limit(limit)
            ).all()
        )

    def add_turn(self, turn: ConversationTurn) -> ConversationTurn:
        self.db.add(turn)
        self.db.flush()
        return turn

    def count_turns(self, session_id: str) -> int:
        result = self.db.scalar(
            select(func.count(ConversationTurn.id)).where(ConversationTurn.session_id == session_id)
        )
        return result or 0

    def clear_turns(self, session_id: str) -> None:
        self.db.execute(delete(ConversationTurn).where(ConversationTurn.session_id == session_id))

    def clear_expired(self, ttl_minutes: int) -> None:
        from datetime import datetime, timedelta

        cutoff = datetime.utcnow() - timedelta(minutes=ttl_minutes)
        self.db.execute(delete(ConversationTurn).where(ConversationTurn.created_at < cutoff))
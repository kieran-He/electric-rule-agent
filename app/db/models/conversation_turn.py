from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ConversationTurn(Base):
    __tablename__ = "conversation_turn"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    turn_index: Mapped[int] = mapped_column(Integer)
    user_query: Mapped[str] = mapped_column(Text)
    bot_reply: Mapped[str] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    province_code: Mapped[str | None] = mapped_column(String(8), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_conversation_turn_session_turn", "session_id", "turn_index"),)
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ConversationState(Base):
    __tablename__ = "conversation_state"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    province_code: Mapped[str | None] = mapped_column(String(8), nullable=True)
    market_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entity_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    time_scope: Mapped[str | None] = mapped_column(String(64), nullable=True)
    current_doc_scope: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    history_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(String(128), nullable=True)
    title_generated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

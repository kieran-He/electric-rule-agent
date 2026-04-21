from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class TraceRecord(Base):
    __tablename__ = "trace_record"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    raw_query: Mapped[str] = mapped_column(Text)
    rewritten_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    filters: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_doc_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    rerank_scores: Mapped[str | None] = mapped_column(Text, nullable=True)
    used_clause_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_doc_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

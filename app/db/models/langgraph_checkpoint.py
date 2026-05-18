from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class LangGraphCheckpoint(Base):
    __tablename__ = "langgraph_checkpoint"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(128), index=True)
    checkpoint_ns: Mapped[str] = mapped_column(String(64), default="")
    checkpoint_id: Mapped[str] = mapped_column(String(64))
    parent_checkpoint_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checkpoint_data: Mapped[str] = mapped_column(Text)
    checkpoint_metadata: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_checkpoint_thread_ns", "thread_id", "checkpoint_ns"),
        UniqueConstraint("thread_id", "checkpoint_ns", "checkpoint_id", name="uq_checkpoint_thread_ns_id"),
    )
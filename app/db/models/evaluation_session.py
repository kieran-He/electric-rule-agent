from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class EvaluationSession(Base):
    __tablename__ = "evaluation_session"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    eval_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    benchmark_version: Mapped[str] = mapped_column(String(32), default="v1.0")
    total_questions: Mapped[int] = mapped_column(Integer, default=0)
    pass_count: Mapped[int] = mapped_column(Integer, default=0)
    overall_pass: Mapped[bool] = mapped_column(Boolean, default=False)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)
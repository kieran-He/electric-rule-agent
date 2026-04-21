from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class EvaluationRecord(Base):
    __tablename__ = "evaluation_record"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(1024))
    expected_doc: Mapped[str | None] = mapped_column(String(512), nullable=True)
    expected_article: Mapped[str | None] = mapped_column(String(64), nullable=True)
    predicted_doc: Mapped[str | None] = mapped_column(String(512), nullable=True)
    predicted_article: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    trace_id: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True, default="clause_qa")
    eval_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    benchmark_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    expected_keywords_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_faithfulness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_answer_relevancy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_context_precision_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    question_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

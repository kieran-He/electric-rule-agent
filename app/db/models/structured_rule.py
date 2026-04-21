from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class StructuredRule(Base):
    __tablename__ = "structured_rule"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    clause_id: Mapped[int] = mapped_column(ForeignKey("clause.id"), index=True)
    subject: Mapped[str | None] = mapped_column(String(128), nullable=True)
    predicate: Mapped[str | None] = mapped_column(String(128), nullable=True)
    object_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    condition_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    numeric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    numeric_unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    time_value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    time_unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    clause = relationship("Clause", back_populates="structured_rules")

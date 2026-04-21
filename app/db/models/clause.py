from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Clause(Base):
    __tablename__ = "clause"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("document.id"), index=True)
    chapter_no: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chapter_title: Mapped[str | None] = mapped_column(String(256), nullable=True)
    section_no: Mapped[str | None] = mapped_column(String(64), nullable=True)
    section_title: Mapped[str | None] = mapped_column(String(256), nullable=True)
    article_no: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    item_no: Mapped[str | None] = mapped_column(String(64), nullable=True)
    title_path: Mapped[str] = mapped_column(String(1024), index=True)
    clause_text: Mapped[str] = mapped_column(Text)
    clause_summary: Mapped[str | None] = mapped_column(String(512), nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", back_populates="clauses")
    rule_tags = relationship("RuleTag", back_populates="clause", cascade="all, delete-orphan")
    structured_rules = relationship("StructuredRule", back_populates="clause", cascade="all, delete-orphan")

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Document(Base):
    __tablename__ = "document"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    province_code: Mapped[str] = mapped_column(String(8), default="SN", index=True)
    doc_name: Mapped[str] = mapped_column(String(512), index=True)
    doc_type: Mapped[str] = mapped_column(String(64), default="notice", index=True)
    market_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    subject_scope: Mapped[str | None] = mapped_column(String(256), nullable=True)
    version_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="formal", index=True)
    issuer: Mapped[str | None] = mapped_column(String(512), nullable=True)
    issue_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    effective_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    source_file: Mapped[str] = mapped_column(String(1024), unique=True)
    file_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    parent_doc_id: Mapped[int | None] = mapped_column(ForeignKey("document.id"), nullable=True)
    issuer: Mapped[str | None] = mapped_column(String(256), nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    parent = relationship("Document", remote_side=[id], backref="children")
    clauses = relationship("Clause", back_populates="document", cascade="all, delete-orphan")

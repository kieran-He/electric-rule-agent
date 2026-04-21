from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class RuleTag(Base):
    __tablename__ = "rule_tag"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    clause_id: Mapped[int] = mapped_column(ForeignKey("clause.id"), index=True)
    province_code: Mapped[str] = mapped_column(String(8), default="SN", index=True)
    market_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    entity_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    trade_cycle: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trade_mode: Mapped[str | None] = mapped_column(String(64), nullable=True)
    time_granularity: Mapped[str | None] = mapped_column(String(64), nullable=True)
    action_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    price_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    settlement_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metering_required: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    access_condition: Mapped[str | None] = mapped_column(String(256), nullable=True)
    penalty_related: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    green_power_related: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    spot_related: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    retail_related: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    storage_related: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    vpp_related: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    clause = relationship("Clause", back_populates="rule_tags")

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.base import Base


engine = create_engine(
    settings.database_url,
    future=True,
    echo=False,
    connect_args={"check_same_thread": False} if settings.is_sqlite else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    from app.db.models import clause, conversation_state, document, evaluation_record, evaluation_session, rule_tag, structured_rule, trace_record  # noqa: F401

    Base.metadata.create_all(bind=engine)

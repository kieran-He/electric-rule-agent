from __future__ import annotations

from sqlalchemy import create_engine, text
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
    from app.db.models import (  # noqa: F401
        clause,
        conversation_state,
        conversation_turn,
        document,
        evaluation_record,
        evaluation_session,
        metrics_record,
        processed_message,
        rule_tag,
        structured_rule,
        trace_record,
        user_feedback,
    )

    with engine.connect() as conn:
        try:
            conn.execute(text("SELECT event_id FROM processed_messages LIMIT 1"))
        except Exception:
            conn.execute(text("DROP TABLE IF EXISTS processed_messages"))
            conn.commit()

    Base.metadata.create_all(bind=engine)

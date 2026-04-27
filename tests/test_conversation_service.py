from __future__ import annotations

import pytest

from app.db.session import SessionLocal, init_db
from app.services.conversation_service import ConversationService


@pytest.fixture
def conversation_service():
    init_db()
    return ConversationService(session_factory=SessionLocal)


def test_get_or_create(conversation_service):
    state = conversation_service.get_or_create("test_session_1")
    assert state.session_id == "test_session_1"


def test_append_turn(conversation_service):
    from sqlalchemy import text
    from app.db.session import SessionLocal
    
    # Clean up test data before test
    with SessionLocal() as db:
        db.execute(text("DELETE FROM conversation_turn WHERE session_id = :sid"), {"sid": "test_session_2"})
        db.execute(text("DELETE FROM conversation_state WHERE session_id = :sid"), {"sid": "test_session_2"})
        db.commit()
    
    conversation_service.append_turn(
        session_id="test_session_2",
        user_query="陕西交易规则",
        bot_reply="根据规则...",
        intent="clause_qa"
    )
    history = conversation_service.get_history("test_session_2")
    assert len(history) == 2
    assert history[0] == "Q: 陕西交易规则"
    assert history[1] == "A: 根据规则..."


def test_update_context(conversation_service):
    from sqlalchemy import text
    from app.db.session import SessionLocal
    
    # Clean up test data before test
    with SessionLocal() as db:
        db.execute(text("DELETE FROM conversation_state WHERE session_id = :sid"), {"sid": "test_session_3"})
        db.commit()
    
    conversation_service.update_context(
        session_id="test_session_3",
        province_code="GD"
    )
    state = conversation_service.get_or_create("test_session_3")
    assert state.province_code == "GD"


def test_get_history_empty(conversation_service):
    history = conversation_service.get_history("test_session_new")
    assert history == []
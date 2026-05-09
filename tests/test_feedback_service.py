from __future__ import annotations

import pytest
from sqlalchemy import text

from app.db.session import SessionLocal, init_db
from app.services.feedback_service import FeedbackService
from app.schemas.feedback import FeedbackRequest


@pytest.fixture
def feedback_service():
    init_db()
    return FeedbackService(session_factory=SessionLocal)


def test_submit_feedback(feedback_service):
    with SessionLocal() as db:
        db.execute(text("DELETE FROM user_feedback WHERE trace_id = :tid"), {"tid": "trace_test_fb_001"})
        db.commit()
    
    req = FeedbackRequest(
        trace_id="trace_test_fb_001",
        session_id="session_fb_001",
        query="测试查询",
        answer="测试答案",
        rating=5,
        feedback_type="helpful",
        user_comment="很好",
        suggested_correction=None
    )
    
    result = feedback_service.submit_feedback(req)
    assert result.trace_id == "trace_test_fb_001"
    assert result.rating == 5
    assert result.feedback_type == "helpful"


def test_get_feedback_by_trace(feedback_service):
    with SessionLocal() as db:
        db.execute(text("DELETE FROM user_feedback WHERE trace_id = :tid"), {"tid": "trace_test_fb_002"})
        db.commit()
    
    req = FeedbackRequest(
        trace_id="trace_test_fb_002",
        session_id="session_fb_002",
        query="测试查询2",
        answer="测试答案2",
        rating=3,
        feedback_type="incomplete"
    )
    
    feedback_service.submit_feedback(req)
    
    results = feedback_service.get_feedback_by_trace("trace_test_fb_002")
    assert len(results) >= 1
    assert results[0].rating == 3
    assert results[0].feedback_type == "incomplete"


def test_get_feedback_stats(feedback_service):
    with SessionLocal() as db:
        db.execute(text("DELETE FROM user_feedback WHERE session_id = :sid"), {"sid": "session_fb_stats"})
        db.commit()
    
    for i in range(3):
        req = FeedbackRequest(
            trace_id=f"trace_test_fb_stats_{i}",
            session_id="session_fb_stats",
            rating=i + 3,
            feedback_type="helpful" if i % 2 == 0 else "wrong"
        )
        feedback_service.submit_feedback(req)
    
    stats = feedback_service.get_feedback_stats("session_fb_stats")
    assert stats.total_feedback == 3
    assert stats.avg_rating == 4.0
    assert stats.rating_distribution.get(3) == 1
    assert stats.rating_distribution.get(4) == 1
    assert stats.rating_distribution.get(5) == 1


def test_feedback_validation(feedback_service):
    import pydantic
    
    with pytest.raises(pydantic.ValidationError):
        FeedbackRequest(
            trace_id="trace_test_fb_invalid",
            rating=6,
            feedback_type="helpful"
        )
    
    valid_req = FeedbackRequest(
        trace_id="trace_test_fb_valid",
        rating=3,
        feedback_type="wrong"
    )
    assert valid_req.rating == 3
    assert valid_req.feedback_type == "wrong"
from __future__ import annotations

import pytest

from app.db.session import SessionLocal, init_db
from app.services.trace_service import TraceService


@pytest.fixture
def trace_service():
    init_db()
    return TraceService(session_factory=SessionLocal)


def test_save_trace(trace_service):
    from app.db.session import SessionLocal
    from sqlalchemy import text
    
    # Clean up test data before test
    with SessionLocal() as db:
        db.execute(text("DELETE FROM trace_record WHERE trace_id = :tid"), {"tid": "trace_test_001"})
        db.commit()
    
    trace = trace_service.save_trace(
        trace_id="trace_test_001",
        session_id="session_001",
        raw_query="测试查询",
        latency_ms=100
    )
    # Access attributes before session closes
    trace_id = trace.trace_id
    assert trace_id == "trace_test_001"


def test_get_trace(trace_service):
    from app.db.session import SessionLocal
    from sqlalchemy import text
    
    # Clean up test data before test
    with SessionLocal() as db:
        db.execute(text("DELETE FROM trace_record WHERE trace_id = :tid"), {"tid": "trace_test_002"})
        db.commit()
    
    trace_service.save_trace(
        trace_id="trace_test_002",
        session_id="session_002",
        raw_query="测试查询2",
        latency_ms=200
    )
    
    retrieved = trace_service.get_trace("trace_test_002")
    assert retrieved is not None
    assert retrieved.raw_query == "测试查询2"
    assert retrieved.latency_ms == 200


def test_get_trace_not_found(trace_service):
    retrieved = trace_service.get_trace("trace_not_exist")
    assert retrieved is None


def test_save_trace_with_rerank_scores(trace_service):
    from app.db.session import SessionLocal
    from sqlalchemy import text
    
    # Clean up test data before test
    with SessionLocal() as db:
        db.execute(text("DELETE FROM trace_record WHERE trace_id = :tid"), {"tid": "trace_test_003"})
        db.commit()
    
    trace = trace_service.save_trace(
        trace_id="trace_test_003",
        session_id="session_003",
        raw_query="测试查询3",
        rerank_scores=[0.85, 0.72, 0.65],
        latency_ms=150
    )
    # Access attributes before session closes
    trace_id = trace.trace_id
    assert trace_id == "trace_test_003"
    
    retrieved = trace_service.get_trace("trace_test_003")
    assert retrieved.rerank_scores == [0.85, 0.72, 0.65]
from __future__ import annotations

from sqlalchemy import text

from app.core.metrics import MetricsStore, metrics_store
from app.db.session import SessionLocal, init_db
from app.db.repositories.metrics_repo import MetricsRepository


def test_record_latency():
    store = MetricsStore()
    store.record_latency(100.0, "total")
    store.record_latency(50.0, "retrieval")
    store.record_latency(30.0, "llm")
    
    summary = store.get_summary()
    assert summary["latency_avg_ms"] == 100.0
    assert summary["retrieval_latency_avg_ms"] == 50.0
    assert summary["llm_latency_avg_ms"] == 30.0


def test_record_query():
    store = MetricsStore()
    store.record_query("SN")
    store.record_query("SN")
    store.record_query("GD")
    
    summary = store.get_summary()
    assert summary["query_counts"]["SN"] == 2
    assert summary["query_counts"]["GD"] == 1


def test_record_error():
    store = MetricsStore()
    store.record_error("llm_timeout")
    store.record_error("llm_timeout")
    store.record_error("db_error")
    
    summary = store.get_summary()
    assert summary["error_counts"]["llm_timeout"] == 2
    assert summary["error_counts"]["db_error"] == 1


def test_clear():
    store = MetricsStore()
    store.record_latency(100.0, "total")
    store.record_query("SN")
    
    store.clear()
    
    summary = store.get_summary()
    assert summary["latency_avg_ms"] == 0.0
    assert summary["query_counts"] == {}


def test_avg_with_many_samples():
    store = MetricsStore()
    for i in range(150):
        store.record_latency(float(i), "total")
    
    summary = store.get_summary()
    avg = summary["latency_avg_ms"]
    expected_avg = sum(range(50, 150)) / 100
    assert avg == expected_avg


def test_save_to_db():
    init_db()
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM metrics_record WHERE trace_id = :tid"), {"tid": "test_save_to_db_001"})
        db.commit()
    finally:
        db.close()
    
    store = MetricsStore()
    store.save_to_db(
        db=SessionLocal(),
        trace_id="test_save_to_db_001",
        session_id="session_save_test",
        request_id="request_save_test",
        user_id="user_save_test",
        retrieval_latency_ms=100,
        llm_latency_ms=200,
        total_latency_ms=300,
        input_tokens=50,
        output_tokens=100,
        province_code="SN",
        success=True,
    )
    
    repo = MetricsRepository(SessionLocal())
    record = repo.get_by_trace_id("test_save_to_db_001")
    assert record is not None
    assert record.retrieval_latency_ms == 100
    assert record.llm_latency_ms == 200
    assert record.total_latency_ms == 300
    assert record.input_tokens == 50
    assert record.output_tokens == 100
    assert record.province_code == "SN"
    assert record.session_id == "session_save_test"
    assert record.request_id == "request_save_test"
    assert record.user_id == "user_save_test"
    assert record.success is True


def test_save_to_db_with_error():
    init_db()
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM metrics_record WHERE trace_id = :tid"), {"tid": "test_save_error_001"})
        db.commit()
    finally:
        db.close()
    
    store = MetricsStore()
    store.save_to_db(
        db=SessionLocal(),
        trace_id="test_save_error_001",
        session_id="session_error_test",
        error_type="llm_timeout",
        error_message="Timeout after 30s",
        success=False,
    )
    
    repo = MetricsRepository(SessionLocal())
    record = repo.get_by_trace_id("test_save_error_001")
    assert record is not None
    assert record.success is False
    assert record.error_type == "llm_timeout"
    assert record.error_message == "Timeout after 30s"
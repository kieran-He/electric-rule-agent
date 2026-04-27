from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from sqlalchemy import text

from app.db.session import SessionLocal, init_db
from app.db.repositories.metrics_repo import MetricsRepository
from app.db.models.metrics_record import MetricsRecord


@pytest.fixture
def metrics_repo():
    init_db()
    return MetricsRepository(SessionLocal())


def test_get_by_trace_id(metrics_repo):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM metrics_record WHERE trace_id = :tid"), {"tid": "test_trace_repo_001"})
        db.commit()
        
        record = MetricsRecord(
            trace_id="test_trace_repo_001",
            session_id="session_001",
            retrieval_latency_ms=100,
            llm_latency_ms=200,
            total_latency_ms=300,
            input_tokens=50,
            output_tokens=100,
            province_code="SN",
            success=True,
        )
        db.add(record)
        db.commit()
        
        retrieved = metrics_repo.get_by_trace_id("test_trace_repo_001")
        assert retrieved is not None
        assert retrieved.retrieval_latency_ms == 100
        assert retrieved.llm_latency_ms == 200
    finally:
        db.close()


def test_get_recent(metrics_repo):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM metrics_record WHERE trace_id LIKE 'test_recent_%'"))
        db.commit()
        
        for i in range(5):
            record = MetricsRecord(
                trace_id=f"test_recent_{i:03d}",
                session_id="session_recent",
                total_latency_ms=100 + i,
                success=True,
            )
            db.add(record)
        db.commit()
        
        records = metrics_repo.get_recent(limit=3)
        assert len(records) == 3
        assert all(r.trace_id.startswith("test_recent_") for r in records)
    finally:
        db.close()


def test_get_summary(metrics_repo):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM metrics_record WHERE trace_id LIKE 'test_summary_%'"))
        db.commit()
        
        cutoff = datetime.utcnow() - timedelta(hours=1)
        
        for i in range(3):
            record = MetricsRecord(
                trace_id=f"test_summary_{i:03d}",
                session_id="session_summary",
                retrieval_latency_ms=100,
                llm_latency_ms=200,
                total_latency_ms=300,
                input_tokens=50,
                output_tokens=100,
                success=True,
                created_at=cutoff + timedelta(minutes=i * 10),
            )
            db.add(record)
        db.commit()
        
        summary = metrics_repo.get_summary(hours=24)
        assert summary["count"] >= 3
        assert summary["avg_total_latency_ms"] > 0
        assert summary["total_tokens"] >= 450
    finally:
        db.close()


def test_get_error_summary(metrics_repo):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM metrics_record WHERE trace_id LIKE 'test_error_%'"))
        db.commit()
        
        cutoff = datetime.utcnow() - timedelta(hours=1)
        
        record_success = MetricsRecord(
            trace_id="test_error_success",
            session_id="session_error",
            success=True,
            created_at=cutoff,
        )
        record_fail = MetricsRecord(
            trace_id="test_error_fail",
            session_id="session_error",
            success=False,
            error_type="llm_error",
            error_message="Timeout",
            created_at=cutoff,
        )
        db.add_all([record_success, record_fail])
        db.commit()
        
        summary = metrics_repo.get_error_summary(hours=24)
        assert summary["error_count"] >= 1
    finally:
        db.close()


def test_get_province_stats(metrics_repo):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM metrics_record WHERE trace_id LIKE 'test_province_%'"))
        db.commit()
        
        cutoff = datetime.utcnow() - timedelta(hours=1)
        
        for i in range(3):
            record = MetricsRecord(
                trace_id=f"test_province_sn_{i:03d}",
                session_id="session_province",
                province_code="SN",
                success=True,
                created_at=cutoff,
            )
            db.add(record)
        
        for i in range(2):
            record = MetricsRecord(
                trace_id=f"test_province_gd_{i:03d}",
                session_id="session_province",
                province_code="GD",
                success=True,
                created_at=cutoff,
            )
            db.add(record)
        db.commit()
        
        stats = metrics_repo.get_province_stats(hours=24)
        assert "SN" in stats
        assert stats["SN"] >= 3
        assert stats["GD"] >= 2
    finally:
        db.close()


def test_get_by_session(metrics_repo):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM metrics_record WHERE trace_id LIKE 'test_session_%'"))
        db.commit()
        
        for i in range(3):
            record = MetricsRecord(
                trace_id=f"test_session_{i:03d}",
                session_id="test_session_abc",
                success=True,
            )
            db.add(record)
        db.commit()
        
        records = metrics_repo.get_by_session("test_session_abc", limit=10)
        assert len(records) >= 3
    finally:
        db.close()


def test_get_by_user(metrics_repo):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM metrics_record WHERE trace_id LIKE 'test_user_%'"))
        db.commit()
        
        for i in range(2):
            record = MetricsRecord(
                trace_id=f"test_user_{i:03d}",
                session_id="session_user",
                user_id="test_user_123",
                success=True,
            )
            db.add(record)
        db.commit()
        
        records = metrics_repo.get_by_user("test_user_123", limit=10)
        assert len(records) >= 2
    finally:
        db.close()
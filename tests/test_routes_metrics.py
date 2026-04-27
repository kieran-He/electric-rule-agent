from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from sqlalchemy import text

from app.db.session import SessionLocal, init_db
from app.db.models.metrics_record import MetricsRecord


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    init_db()
    return TestClient(app)


def test_get_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "latency_avg_ms" in data
    assert "retrieval_latency_avg_ms" in data
    assert "llm_latency_avg_ms" in data
    assert "query_counts" in data
    assert "error_counts" in data


def test_get_metrics_health(client):
    response = client.get("/metrics/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "reranker_loaded" in data
    assert "metrics_samples" in data


def test_get_metrics_history(client):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM metrics_record WHERE trace_id LIKE 'test_api_history_%'"))
        db.commit()
        
        cutoff = datetime.utcnow() - timedelta(hours=1)
        for i in range(3):
            record = MetricsRecord(
                trace_id=f"test_api_history_{i:03d}",
                session_id="session_api",
                retrieval_latency_ms=100 + i,
                llm_latency_ms=200 + i,
                total_latency_ms=300 + i,
                input_tokens=50,
                output_tokens=100,
                success=True,
                created_at=cutoff,
            )
            db.add(record)
        db.commit()
    finally:
        db.close()
    
    response = client.get("/metrics/history?hours=24")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "avg_retrieval_latency_ms" in data
    assert "avg_llm_latency_ms" in data
    assert "total_tokens" in data


def test_get_metrics_errors(client):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM metrics_record WHERE trace_id LIKE 'test_api_error_%'"))
        db.commit()
        
        cutoff = datetime.utcnow() - timedelta(hours=1)
        record = MetricsRecord(
            trace_id="test_api_error_001",
            session_id="session_error",
            success=False,
            error_type="llm_timeout",
            error_message="Timeout after 30s",
            created_at=cutoff,
        )
        db.add(record)
        db.commit()
    finally:
        db.close()
    
    response = client.get("/metrics/errors?hours=24")
    assert response.status_code == 200
    data = response.json()
    assert "error_count" in data


def test_get_metrics_province(client):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM metrics_record WHERE trace_id LIKE 'test_api_province_%'"))
        db.commit()
        
        cutoff = datetime.utcnow() - timedelta(hours=1)
        for i in range(2):
            record = MetricsRecord(
                trace_id=f"test_api_province_sn_{i:03d}",
                session_id="session_province",
                province_code="SN",
                success=True,
                created_at=cutoff,
            )
            db.add(record)
        for i in range(3):
            record = MetricsRecord(
                trace_id=f"test_api_province_gd_{i:03d}",
                session_id="session_province",
                province_code="GD",
                success=True,
                created_at=cutoff,
            )
            db.add(record)
        db.commit()
    finally:
        db.close()
    
    response = client.get("/metrics/province?hours=24")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_get_metrics_recent(client):
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM metrics_record WHERE trace_id LIKE 'test_api_recent_%'"))
        db.commit()
        
        for i in range(5):
            record = MetricsRecord(
                trace_id=f"test_api_recent_{i:03d}",
                session_id="session_recent",
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
    finally:
        db.close()
    
    response = client.get("/metrics/recent?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        first = data[0]
        assert "trace_id" in first
        assert "session_id" in first
        assert "retrieval_latency_ms" in first
        assert "llm_latency_ms" in first
        assert "total_latency_ms" in first
        assert "input_tokens" in first
        assert "output_tokens" in first
        assert "province_code" in first
        assert "success" in first
        assert "created_at" in first
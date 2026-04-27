import pytest
from fastapi.testclient import TestClient

from app import main


client = TestClient(main.app)


@pytest.mark.skip(reason="ingest_enabled setting removed from config")
def test_ingest_rejected_when_ingest_disabled(monkeypatch):
    pass


@pytest.mark.skip(reason="ingest_enabled setting removed from config")
def test_ingest_allowed_when_ingest_enabled(monkeypatch):
    pass


def test_health_endpoint():
    resp = client.get("/metrics/health")
    assert resp.status_code in [200, 503]
    data = resp.json()
    assert "status" in data


def test_query_endpoint_returns_422_on_empty_query():
    resp = client.post(
        "/query",
        json={
            "query": "",
            "session_id": "test-session",
            "top_k": 5,
        },
    )
    assert resp.status_code == 422
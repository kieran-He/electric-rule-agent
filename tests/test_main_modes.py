from fastapi.testclient import TestClient

from app import main


client = TestClient(main.app)


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
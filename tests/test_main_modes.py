from fastapi.testclient import TestClient

from app import main
from app.generator import LLMGenerationError
from app.ingestion import IngestStats
from app.schemas import QueryMode, QueryResponse


client = TestClient(main.app)


def test_ingest_rejected_when_ingest_disabled(monkeypatch):
    monkeypatch.setattr(main.settings, "ingest_enabled", False)
    resp = client.post(
        "/admin/ingest",
        json={
            "kb_scope": "province",
            "province_code": "SN",
            "cleaning_profile": "robust",
        },
    )
    assert resp.status_code == 403
    assert "disabled in online mode" in resp.json()["detail"]


def test_ingest_allowed_when_ingest_enabled(monkeypatch):
    monkeypatch.setattr(main.settings, "ingest_enabled", True)

    def fake_ingest_path(**_kwargs):
        return IngestStats(
            files_processed=1,
            chunks_created=3,
            files_new=1,
            files_updated=0,
            files_skipped=0,
            ocr_pages_processed=0,
        )

    monkeypatch.setattr(main.ingestor, "ingest_path", fake_ingest_path)
    resp = client.post(
        "/admin/ingest",
        json={
            "kb_scope": "province",
            "province_code": "SN",
            "cleaning_profile": "robust",
            "dedupe": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["chunks_created"] == 3


def test_query_works_when_ingest_disabled(monkeypatch):
    monkeypatch.setattr(main.settings, "ingest_enabled", False)

    def fake_process(_req):
        return QueryResponse(
            mode=QueryMode.province_plus_global,
            province_code="SN",
            conclusion="ok",
            follow_up="next",
        )

    monkeypatch.setattr(main.service, "process", fake_process)
    resp = client.post(
        "/query",
        json={
            "query": "陕西中长期交易流程",
            "session_id": "test-session",
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["conclusion"] == "ok"


def test_query_returns_503_when_llm_fails(monkeypatch):
    def fake_process(_req):
        raise LLMGenerationError("upstream timeout")

    monkeypatch.setattr(main.service, "process", fake_process)
    resp = client.post(
        "/query",
        json={
            "query": "陕西中长期交易流程",
            "session_id": "test-session-2",
            "top_k": 5,
        },
    )
    assert resp.status_code == 503
    assert "upstream timeout" in resp.json()["detail"]

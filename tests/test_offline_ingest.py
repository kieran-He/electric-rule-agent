from app.services.ingest.document_ingestor import IngestStats
from tools import offline_ingest


class FakeRepo:
    def __init__(self, persist_directory, embedding_model_name):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.ready = True
        self.init_error = None


class FakeIngestor:
    def __init__(self, repository, index_path):
        self.repository = repository
        self.index_path = index_path

    def ingest_path(self, **kwargs):
        return IngestStats(
            files_processed=2,
            chunks_created=6,
            files_new=1,
            files_updated=1,
            files_skipped=0,
            ocr_pages_processed=3,
        )


def test_run_offline_ingest_success(monkeypatch):
    monkeypatch.setattr(offline_ingest, "ChromaPolicyRepository", FakeRepo)
    monkeypatch.setattr(offline_ingest, "DocumentIngestor", FakeIngestor)

    result = offline_ingest.run_offline_ingest(
        kb_scope="province",
        province_code="SN",
        docs_path="data/docs/SN",
        docs_root=None,
        rebuild=False,
        dedupe=True,
        enable_ocr=False,
        chunk_size=800,
        chunk_overlap=120,
    )
    assert result["success"] is True
    assert result["chunks_created"] == 6
    assert result["kb_scope"] == "province"


def test_run_offline_ingest_requires_province_code():
    try:
        offline_ingest.run_offline_ingest(
            kb_scope="province",
            province_code=None,
            docs_path=None,
            docs_root=None,
            rebuild=False,
            dedupe=True,
            enable_ocr=False,
            chunk_size=800,
            chunk_overlap=120,
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "province_code is required" in str(exc)


def test_run_offline_ingest_rejects_bad_chunk_overlap():
    try:
        offline_ingest.run_offline_ingest(
            kb_scope="global",
            province_code=None,
            docs_path="data/docs",
            docs_root=None,
            rebuild=False,
            dedupe=True,
            enable_ocr=False,
            chunk_size=100,
            chunk_overlap=100,
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "chunk_overlap must be smaller" in str(exc)

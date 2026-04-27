import shutil
import uuid
from pathlib import Path

from app.services.ingest.document_ingestor import DocumentIngestor


class FakeRepo:
    def __init__(self):
        self.ingested = []
        self.deleted_hashes = []

    def ingest_chunks(self, texts, metadatas, kb_scope, province_code, rebuild=False):
        self.ingested.append(
            {
                "texts": texts,
                "metadatas": metadatas,
                "kb_scope": kb_scope,
                "province_code": province_code,
                "rebuild": rebuild,
            }
        )
        return len(texts)

    def delete_by_file_hash(self, kb_scope, province_code, file_hash):
        self.deleted_hashes.append((kb_scope, province_code, file_hash))


def _make_workspace() -> Path:
    base = Path("tests/.tmp") / f"ingest-{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _build_ingestor(workspace: Path, repo: FakeRepo) -> DocumentIngestor:
    return DocumentIngestor(repo, index_path=str(workspace / "ingest_index.json"))


def test_ingestion_enriches_metadata():
    workspace = _make_workspace()
    docs = workspace / "docs"
    docs.mkdir()
    file = docs / "sample.txt"
    file.write_text(
        "陕西省电力市场规则，发布日期2026年1月2日。交易组织按照月度开展，经营主体应按时申报并履约。",
        encoding="utf-8",
    )

    repo = FakeRepo()
    try:
        ingestor = _build_ingestor(workspace, repo)
        stats = ingestor.ingest_path(
            docs_path=str(docs),
            kb_scope="province",
            province_code="SN",
            rebuild=False,
            chunk_size=200,
            chunk_overlap=30,
            enable_ocr=False,
            dedupe=True,
            min_ch_ratio=0.05,
            max_replacement_ratio=0.03,
            empty_page_threshold=0.3,
        )

        assert stats.files_processed == 1
        assert stats.files_new == 1
        assert stats.chunks_created > 0
        assert len(repo.ingested) == 1
        metadata = repo.ingested[0]["metadatas"][0]
        assert metadata["doc_title"] == "sample"
        assert metadata["effective_date"] == "2026-01-02"
        assert metadata["policy_level"] in {"province", "unknown"}
        assert metadata["doc_id"].count(":") == 1
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_dedupe_skips_unchanged_file():
    workspace = _make_workspace()
    docs = workspace / "docs"
    docs.mkdir()
    file = docs / "repeat.txt"
    file.write_text("陕西市场交易规则", encoding="utf-8")

    repo = FakeRepo()
    try:
        ingestor = _build_ingestor(workspace, repo)
        ingestor.ingest_path(
            docs_path=str(docs),
            kb_scope="province",
            province_code="SN",
            rebuild=False,
            chunk_size=200,
            chunk_overlap=30,
            enable_ocr=False,
            dedupe=True,
            min_ch_ratio=0.05,
            max_replacement_ratio=0.03,
            empty_page_threshold=0.3,
        )
        second = ingestor.ingest_path(
            docs_path=str(docs),
            kb_scope="province",
            province_code="SN",
            rebuild=False,
            chunk_size=200,
            chunk_overlap=30,
            enable_ocr=False,
            dedupe=True,
            min_ch_ratio=0.05,
            max_replacement_ratio=0.03,
            empty_page_threshold=0.3,
        )

        assert second.files_skipped == 1
        assert second.chunks_created == 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_dedupe_updates_changed_file():
    workspace = _make_workspace()
    docs = workspace / "docs"
    docs.mkdir()
    file = docs / "update.txt"
    file.write_text("版本一", encoding="utf-8")

    repo = FakeRepo()
    try:
        ingestor = _build_ingestor(workspace, repo)
        ingestor.ingest_path(
            docs_path=str(docs),
            kb_scope="province",
            province_code="SN",
            rebuild=False,
            chunk_size=200,
            chunk_overlap=30,
            enable_ocr=False,
            dedupe=True,
            min_ch_ratio=0.05,
            max_replacement_ratio=0.03,
            empty_page_threshold=0.3,
        )

        file.write_text("版本二，内容有变化", encoding="utf-8")
        updated = ingestor.ingest_path(
            docs_path=str(docs),
            kb_scope="province",
            province_code="SN",
            rebuild=False,
            chunk_size=200,
            chunk_overlap=30,
            enable_ocr=False,
            dedupe=True,
            min_ch_ratio=0.05,
            max_replacement_ratio=0.03,
            empty_page_threshold=0.3,
        )

        assert updated.files_updated == 1
        assert len(repo.deleted_hashes) == 1
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

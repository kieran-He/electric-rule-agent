from __future__ import annotations

from pathlib import Path
from typing import Callable

from sqlalchemy.orm import Session

from app.core.exceptions import AppError
from app.db.repositories.document_repo import DocumentRepository
from app.schemas.admin import DocumentAdminItem
from app.schemas.ingest import IngestRequest, IngestResponse
from app.services.ingest.ingestion_pipeline import IngestionPipeline


class IngestService:
    def __init__(self, settings, session_factory: Callable[[], Session]):
        self.settings = settings
        self.session_factory = session_factory

    def ingest(self, req: IngestRequest) -> IngestResponse:
        path = Path(req.path)
        if not path.exists():
            raise AppError(code="INGEST_PATH_NOT_FOUND", message=f"路径不存在: {req.path}", status_code=400)
        with self.session_factory() as db:
            pipeline = IngestionPipeline(db=db, settings=self.settings)
            stats = pipeline.ingest_path(path=path, province_code=req.province_code, rebuild_index=req.rebuild_index)
            db.commit()
        return IngestResponse(
            success=True,
            imported_documents=stats["imported_documents"],
            imported_clauses=stats["imported_clauses"],
            skipped_documents=stats["skipped_documents"],
            message="ingest completed",
        )

    def rebuild_index(self) -> int:
        with self.session_factory() as db:
            pipeline = IngestionPipeline(db=db, settings=self.settings)
            count = pipeline.rebuild_vector_index()
            db.commit()
            return count

    def list_documents(self) -> list[DocumentAdminItem]:
        with self.session_factory() as db:
            repo = DocumentRepository(db)
            rows = repo.list_documents()
            return [
                DocumentAdminItem(
                    id=row.id,
                    doc_name=row.doc_name,
                    doc_type=row.doc_type,
                    status=row.status,
                    province_code=row.province_code,
                    issuer=row.issuer,
                    version_name=row.version_name,
                    is_current=row.is_current,
                )
                for row in rows
            ]

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.dependency import get_ingest_service
from app.core.health_check import health_checker
from app.schemas.admin import DocumentAdminItem, RebuildIndexResponse
from app.services.ingest_service import IngestService

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/rebuild-index", response_model=RebuildIndexResponse)
def rebuild_index(service: IngestService = Depends(get_ingest_service)) -> RebuildIndexResponse:
    count = service.rebuild_index()
    return RebuildIndexResponse(success=True, message=f"rebuild finished with {count} clause embeddings")


@router.get("/documents", response_model=list[DocumentAdminItem])
def list_documents(service: IngestService = Depends(get_ingest_service)) -> list[DocumentAdminItem]:
    return service.list_documents()


@router.get("/health")
def health_check() -> dict:
    return health_checker.check_all()

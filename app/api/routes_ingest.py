from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependency import get_ingest_service
from app.core.exceptions import AppError
from app.schemas.ingest import IngestRequest, IngestResponse
from app.services.ingest_service import IngestService

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/documents", response_model=IngestResponse)
def ingest_documents(
    req: IngestRequest, service: IngestService = Depends(get_ingest_service)
) -> IngestResponse:
    try:
        return service.ingest(req)
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

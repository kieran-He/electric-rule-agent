from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependency import get_query_service, get_trace_service
from app.core.exceptions import AppError
from app.schemas.answer import QueryAnswer
from app.schemas.query import QueryRequest, TraceResponse
from app.services.query_service import QueryService
from app.services.trace_service import TraceService

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryAnswer)
def query_route(req: QueryRequest, service: QueryService = Depends(get_query_service)) -> QueryAnswer:
    try:
        return service.answer(req)
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get("/trace/{trace_id}", response_model=TraceResponse)
def trace_route(trace_id: str, service: TraceService = Depends(get_trace_service)) -> TraceResponse:
    trace = service.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace

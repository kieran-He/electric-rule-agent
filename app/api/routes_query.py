from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependency import get_query_service, get_trace_service
from app.core.exceptions import AppError
from app.core.logging_context import set_trace_id, set_session_id
from app.schemas.answer import QueryAnswer
from app.schemas.query import QueryRequest, QueryRequestExample, TraceResponse
from app.services.query_service import QueryService
from app.services.trace_service import TraceService

router = APIRouter(
    prefix="/query",
    tags=["query"],
    responses={
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"},
    }
)


@router.post(
    "",
    response_model=QueryAnswer,
    summary="Query Policy Knowledge Base",
    description="Execute a query against the policy knowledge base with hybrid retrieval.",
    responses={
        200: {
            "description": "Successful query response",
            "content": {
                "application/json": {
                    "example": {
                        "answer": "根据《陕西省电力市场交易实施细则》...",
                        "citations": [{"doc_name": "陕西规则.pdf", "excerpt": "..."}],
                        "intent": "clause_qa",
                        "confidence": 0.85,
                        "trace_id": "trace_abc123"
                    }
                }
            }
        }
    }
)
def query_route(req: QueryRequest = QueryRequestExample, service: QueryService = Depends(get_query_service)) -> QueryAnswer:
    trace_id = f"trace_{uuid.uuid4().hex[:12]}"
    set_trace_id(trace_id)
    set_session_id(req.session_id)
    
    try:
        return service.answer(req)
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get(
    "/trace/{trace_id}",
    response_model=TraceResponse,
    summary="Get Trace Record",
    description="Retrieve the trace record for a specific query execution."
)
def trace_route(trace_id: str, service: TraceService = Depends(get_trace_service)) -> TraceResponse:
    trace = service.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace

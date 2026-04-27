from __future__ import annotations

import json
import logging
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.trace_record import TraceRecord
from app.schemas.query import TraceResponse

logger = logging.getLogger(__name__)


class TraceService:
    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def get_trace(self, trace_id: str) -> TraceResponse | None:
        with self.session_factory() as db:
            row = db.scalar(select(TraceRecord).where(TraceRecord.trace_id == trace_id))
            if row is None:
                return None
            return TraceResponse(
                trace_id=row.trace_id,
                raw_query=row.raw_query,
                rewritten_query=row.rewritten_query,
                intent=row.intent,
                filters=json.loads(row.filters or "{}"),
                retrieved_doc_ids=json.loads(row.retrieved_doc_ids or "[]"),
                rerank_scores=json.loads(row.rerank_scores or "[]"),
                used_clause_ids=json.loads(row.used_clause_ids or "[]"),
                final_doc_ids=json.loads(row.final_doc_ids or "[]"),
                latency_ms=int(row.latency_ms or 0),
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                total_tokens=row.total_tokens,
                retrieval_latency_ms=row.retrieval_latency_ms,
                llm_latency_ms=row.llm_latency_ms,
                success=row.success,
                error_type=row.error_type,
                error_message=row.error_message,
            )

    def save_trace(
        self,
        trace_id: str,
        session_id: str,
        raw_query: str,
        rewritten_query: str | None = None,
        intent: str | None = None,
        filters: dict = None,
        retrieved_doc_ids: list[int] = None,
        rerank_scores: list[float] = None,
        used_clause_ids: list[int] = None,
        final_doc_ids: list[int] = None,
        latency_ms: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        retrieval_latency_ms: int = 0,
        llm_latency_ms: int = 0,
        error_type: str | None = None,
        error_message: str | None = None,
        success: bool = True,
    ) -> TraceRecord:
        with self.session_factory() as db:
            trace = TraceRecord(
                trace_id=trace_id,
                session_id=session_id,
                raw_query=raw_query,
                rewritten_query=rewritten_query,
                intent=intent,
                filters=json.dumps(filters or {}),
                retrieved_doc_ids=json.dumps(retrieved_doc_ids or []),
                rerank_scores=json.dumps(rerank_scores or []),
                used_clause_ids=json.dumps(used_clause_ids or []),
                final_doc_ids=json.dumps(final_doc_ids or []),
                latency_ms=str(latency_ms),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=llm_latency_ms,
                error_type=error_type,
                error_message=error_message,
                success=success,
            )
            db.add(trace)
            db.flush()
            db.refresh(trace)
            _trace_id = trace.trace_id
            _session_id = trace.session_id
            db.commit()
            logger.debug(f"Trace saved: {trace_id}")
            trace.trace_id = _trace_id
            trace.session_id = _session_id
            return trace

from __future__ import annotations

import json
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.trace_record import TraceRecord
from app.schemas.query import TraceResponse


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
            )

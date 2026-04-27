from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models.trace_record import TraceRecord


class TraceRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        return self.db.scalar(select(TraceRecord).where(TraceRecord.trace_id == trace_id))

    def get_traces_by_session(self, session_id: str, limit: int = 100) -> list[TraceRecord]:
        return list(
            self.db.scalars(
                select(TraceRecord)
                .where(TraceRecord.session_id == session_id)
                .order_by(TraceRecord.created_at.desc())
                .limit(limit)
            ).all()
        )

    def add_trace(self, trace: TraceRecord) -> TraceRecord:
        self.db.add(trace)
        self.db.flush()
        return trace

    def clear_expired(self, ttl_minutes: int) -> int:
        cutoff = datetime.utcnow() - timedelta(minutes=ttl_minutes)
        result = self.db.execute(delete(TraceRecord).where(TraceRecord.created_at < cutoff))
        return result.rowcount or 0

    def count_traces(self, session_id: str = None) -> int:
        from sqlalchemy import func
        
        if session_id:
            result = self.db.scalar(
                select(func.count(TraceRecord.id)).where(TraceRecord.session_id == session_id)
            )
        else:
            result = self.db.scalar(select(func.count(TraceRecord.id)))
        return result or 0

    def get_token_stats(self, session_id: str = None) -> dict:
        from sqlalchemy import func
        
        query = select(
            func.sum(TraceRecord.input_tokens).label("total_input"),
            func.sum(TraceRecord.output_tokens).label("total_output"),
            func.sum(TraceRecord.total_tokens).label("total_tokens"),
            func.avg(TraceRecord.latency_ms).label("avg_latency"),
        )
        
        if session_id:
            query = query.where(TraceRecord.session_id == session_id)
        
        result = self.db.execute(query).first()
        
        return {
            "total_input_tokens": result.total_input or 0,
            "total_output_tokens": result.total_output or 0,
            "total_tokens": result.total_tokens or 0,
            "avg_latency_ms": float(result.avg_latency or 0),
        }
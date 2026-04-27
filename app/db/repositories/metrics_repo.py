from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.models.metrics_record import MetricsRecord


class MetricsRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_trace_id(self, trace_id: str) -> MetricsRecord | None:
        return self.db.scalar(select(MetricsRecord).where(MetricsRecord.trace_id == trace_id))
    
    def get_recent(self, limit: int = 100) -> list[MetricsRecord]:
        return list(
            self.db.scalars(
                select(MetricsRecord)
                .order_by(MetricsRecord.created_at.desc())
                .limit(limit)
            ).all()
        )
    
    def get_by_session(self, session_id: str, limit: int = 100) -> list[MetricsRecord]:
        return list(
            self.db.scalars(
                select(MetricsRecord)
                .where(MetricsRecord.session_id == session_id)
                .order_by(MetricsRecord.created_at.desc())
                .limit(limit)
            ).all()
        )
    
    def get_by_user(self, user_id: str, limit: int = 100) -> list[MetricsRecord]:
        return list(
            self.db.scalars(
                select(MetricsRecord)
                .where(MetricsRecord.user_id == user_id)
                .order_by(MetricsRecord.created_at.desc())
                .limit(limit)
            ).all()
        )
    
    def get_summary(self, hours: int = 24) -> dict:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        result = self.db.execute(
            select(
                func.count(MetricsRecord.id).label("count"),
                func.avg(MetricsRecord.retrieval_latency_ms).label("avg_retrieval"),
                func.avg(MetricsRecord.llm_latency_ms).label("avg_llm"),
                func.avg(MetricsRecord.total_latency_ms).label("avg_total"),
                func.sum(MetricsRecord.input_tokens).label("total_input_tokens"),
                func.sum(MetricsRecord.output_tokens).label("total_output_tokens"),
                func.sum(MetricsRecord.input_tokens + MetricsRecord.output_tokens).label("total_tokens"),
            ).where(MetricsRecord.created_at >= cutoff)
        ).first()
        
        return {
            "count": result.count or 0,
            "avg_retrieval_latency_ms": float(result.avg_retrieval or 0),
            "avg_llm_latency_ms": float(result.avg_llm or 0),
            "avg_total_latency_ms": float(result.avg_total or 0),
            "total_input_tokens": result.total_input_tokens or 0,
            "total_output_tokens": result.total_output_tokens or 0,
            "total_tokens": result.total_tokens or 0,
        }
    
    def get_error_summary(self, hours: int = 24) -> dict:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        result = self.db.execute(
            select(
                func.count(MetricsRecord.id).label("error_count"),
            ).where(
                MetricsRecord.created_at >= cutoff,
                MetricsRecord.success.is_(False),
            )
        ).first()
        
        return {
            "error_count": result.error_count or 0,
        }
    
    def get_province_stats(self, hours: int = 24) -> dict:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        result = self.db.execute(
            select(
                MetricsRecord.province_code,
                func.count(MetricsRecord.id).label("count"),
            ).where(MetricsRecord.created_at >= cutoff)
            .group_by(MetricsRecord.province_code)
        ).all()
        
        return {row.province_code: row.count for row in result if row.province_code}
    
    def clear_expired(self, ttl_minutes: int) -> int:
        cutoff = datetime.utcnow() - timedelta(minutes=ttl_minutes)
        result = self.db.execute(delete(MetricsRecord).where(MetricsRecord.created_at < cutoff))
        return result.rowcount or 0
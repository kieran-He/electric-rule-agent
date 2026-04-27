from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.metrics import metrics_store
from app.db.repositories.metrics_repo import MetricsRepository
from app.db.session import SessionLocal

router = APIRouter(prefix="/metrics", tags=["metrics"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("")
def get_metrics() -> dict:
    return metrics_store.get_summary()


@router.get("/health")
def get_health() -> dict:
    from app.langchain.reranker_cache import reranker_cache
    
    return {
        "status": "ok",
        "reranker_loaded": reranker_cache.is_loaded(),
        "metrics_samples": len(metrics_store._latency_samples),
    }


@router.get("/history")
def get_metrics_history(
    hours: int = 24,
    db: Session = Depends(get_db),
) -> dict:
    repo = MetricsRepository(db)
    return repo.get_summary(hours=hours)


@router.get("/errors")
def get_error_summary(
    hours: int = 24,
    db: Session = Depends(get_db),
) -> dict:
    repo = MetricsRepository(db)
    return repo.get_error_summary(hours=hours)


@router.get("/province")
def get_province_stats(
    hours: int = 24,
    db: Session = Depends(get_db),
) -> dict:
    repo = MetricsRepository(db)
    return repo.get_province_stats(hours=hours)


@router.get("/recent")
def get_recent_metrics(
    limit: int = 100,
    db: Session = Depends(get_db),
) -> list:
    repo = MetricsRepository(db)
    records = repo.get_recent(limit=limit)
    return [
        {
            "trace_id": r.trace_id,
            "session_id": r.session_id,
            "request_id": r.request_id,
            "user_id": r.user_id,
            "retrieval_latency_ms": r.retrieval_latency_ms,
            "llm_latency_ms": r.llm_latency_ms,
            "total_latency_ms": r.total_latency_ms,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "province_code": r.province_code,
            "success": r.success,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]
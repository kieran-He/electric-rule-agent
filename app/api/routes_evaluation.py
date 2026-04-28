from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_evaluation_service():
    return EvaluationService(session_factory=SessionLocal)


@router.get("/recent")
def get_recent_evaluations(
    limit: int = 50,
    service: EvaluationService = Depends(get_evaluation_service),
) -> dict:
    """
    Get recent evaluation records with RAGAS scores.
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        List of evaluation records
    """
    records = service.get_recent_evaluations(limit=limit)
    return {"records": records}


@router.get("/summary")
def get_evaluation_summary(
    hours: int = 24,
    service: EvaluationService = Depends(get_evaluation_service),
) -> dict:
    """
    Get evaluation metrics summary.
    
    Args:
        hours: Time window in hours
        
    Returns:
        Summary with average faithfulness, answer_relevancy, context_precision
    """
    return service.get_evaluation_summary(hours=hours)


@router.post("/run")
def run_batch_evaluation(
    batch_size: int = 20,
    service: EvaluationService = Depends(get_evaluation_service),
) -> dict:
    """
    Manually trigger batch evaluation on pending traces.
    
    Args:
        batch_size: Number of traces to evaluate in one batch
        
    Returns:
        Summary of evaluation results
    """
    return service.run_batch_evaluation(batch_size=batch_size)


@router.get("/pending")
def get_pending_traces(
    limit: int = 50,
    service: EvaluationService = Depends(get_evaluation_service),
) -> dict:
    """
    Get traces that have contexts and answer but not yet evaluated.
    
    Args:
        limit: Maximum number of traces to return
        
    Returns:
        List of pending traces
    """
    traces = service.get_pending_traces(limit=limit)
    return {
        "count": len(traces),
        "traces": [
            {
                "trace_id": t.trace_id,
                "raw_query": t.raw_query,
                "has_answer": t.answer_text is not None,
                "has_contexts": t.retrieved_doc_texts is not None,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in traces
        ],
    }
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependency import get_feedback_service
from app.schemas.feedback import FeedbackRequest, FeedbackResponse, FeedbackStatsResponse
from app.services.feedback_service import FeedbackService

router = APIRouter(
    prefix="/feedback",
    tags=["feedback"],
    responses={
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"},
    }
)


@router.post(
    "",
    response_model=FeedbackResponse,
    summary="Submit User Feedback",
    description="Submit user feedback for a query response.",
    responses={
        200: {
            "description": "Feedback submitted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": 1,
                        "trace_id": "trace_abc123",
                        "session_id": "chat_123:user_456",
                        "query": "山东的电力交易规则是什么？",
                        "answer": "根据《山东省电力市场交易规则》...",
                        "rating": 4,
                        "feedback_type": "helpful",
                        "user_comment": "答案很详细",
                        "suggested_correction": None,
                        "created_at": "2024-01-15T10:30:00"
                    }
                }
            }
        }
    }
)
def submit_feedback(
    req: FeedbackRequest,
    service: FeedbackService = Depends(get_feedback_service)
) -> FeedbackResponse:
    if req.rating < 1 or req.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    valid_types = {"helpful", "wrong", "irrelevant", "incomplete"}
    if req.feedback_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"feedback_type must be one of: {valid_types}")
    return service.submit_feedback(req)


@router.get(
    "/trace/{trace_id}",
    response_model=list[FeedbackResponse],
    summary="Get Feedback by Trace ID",
    description="Retrieve all feedback for a specific trace ID."
)
def get_feedback_by_trace(
    trace_id: str,
    service: FeedbackService = Depends(get_feedback_service)
) -> list[FeedbackResponse]:
    return service.get_feedback_by_trace(trace_id)


@router.get(
    "/stats",
    response_model=FeedbackStatsResponse,
    summary="Get Feedback Statistics",
    description="Get aggregated feedback statistics, optionally filtered by session ID."
)
def get_feedback_stats(
    session_id: str | None = None,
    service: FeedbackService = Depends(get_feedback_service)
) -> FeedbackStatsResponse:
    return service.get_feedback_stats(session_id)
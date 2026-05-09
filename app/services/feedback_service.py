from __future__ import annotations

import logging
from typing import Callable

from sqlalchemy.orm import Session

from app.db.models.user_feedback import UserFeedback
from app.db.repositories.feedback_repo import FeedbackRepository
from app.schemas.feedback import FeedbackRequest, FeedbackResponse, FeedbackStatsResponse

logger = logging.getLogger(__name__)


class FeedbackService:
    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def submit_feedback(self, req: FeedbackRequest) -> FeedbackResponse:
        with self.session_factory() as db:
            repo = FeedbackRepository(db)
            feedback = UserFeedback(
                trace_id=req.trace_id,
                session_id=req.session_id,
                query=req.query,
                answer=req.answer,
                rating=req.rating,
                feedback_type=req.feedback_type,
                user_comment=req.user_comment,
                suggested_correction=req.suggested_correction,
            )
            saved = repo.save_feedback(feedback)
            logger.info(f"Feedback saved: trace_id={req.trace_id}, rating={req.rating}, type={req.feedback_type}")
            return FeedbackResponse(
                id=saved.id,
                trace_id=saved.trace_id,
                session_id=saved.session_id,
                query=saved.query,
                answer=saved.answer,
                rating=saved.rating,
                feedback_type=saved.feedback_type,
                user_comment=saved.user_comment,
                suggested_correction=saved.suggested_correction,
                created_at=saved.created_at,
            )

    def get_feedback_by_trace(self, trace_id: str) -> list[FeedbackResponse]:
        with self.session_factory() as db:
            repo = FeedbackRepository(db)
            feedbacks = repo.get_feedback_by_trace(trace_id)
            return [
                FeedbackResponse(
                    id=f.id,
                    trace_id=f.trace_id,
                    session_id=f.session_id,
                    query=f.query,
                    answer=f.answer,
                    rating=f.rating,
                    feedback_type=f.feedback_type,
                    user_comment=f.user_comment,
                    suggested_correction=f.suggested_correction,
                    created_at=f.created_at,
                )
                for f in feedbacks
            ]

    def get_feedback_stats(self, session_id: str | None = None) -> FeedbackStatsResponse:
        with self.session_factory() as db:
            repo = FeedbackRepository(db)
            stats = repo.get_feedback_stats(session_id)
            rating_dist = repo.get_rating_distribution(session_id)
            type_dist = repo.get_feedback_type_distribution(session_id)
            return FeedbackStatsResponse(
                total_feedback=stats["total_feedback"],
                avg_rating=stats["avg_rating"],
                rating_distribution=rating_dist,
                feedback_type_distribution=type_dist,
            )
from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models.user_feedback import UserFeedback


class FeedbackRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_feedback(self, feedback: UserFeedback) -> UserFeedback:
        self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)
        return feedback

    def get_feedback_by_trace(self, trace_id: str) -> list[UserFeedback]:
        return list(
            self.db.scalars(
                select(UserFeedback)
                .where(UserFeedback.trace_id == trace_id)
                .order_by(UserFeedback.created_at.desc())
            ).all()
        )

    def get_feedback_by_session(self, session_id: str, limit: int = 100) -> list[UserFeedback]:
        return list(
            self.db.scalars(
                select(UserFeedback)
                .where(UserFeedback.session_id == session_id)
                .order_by(UserFeedback.created_at.desc())
                .limit(limit)
            ).all()
        )

    def get_feedback_stats(self, session_id: str | None = None) -> dict:
        query = select(
            func.count(UserFeedback.id).label("total"),
            func.avg(UserFeedback.rating).label("avg_rating"),
        )
        if session_id:
            query = query.where(UserFeedback.session_id == session_id)
        result = self.db.execute(query).first()
        return {
            "total_feedback": result.total or 0,
            "avg_rating": float(result.avg_rating or 0),
        }

    def get_rating_distribution(self, session_id: str | None = None) -> dict[int, int]:
        query = select(
            UserFeedback.rating,
            func.count(UserFeedback.id).label("count")
        ).group_by(UserFeedback.rating)
        if session_id:
            query = query.where(UserFeedback.session_id == session_id)
        results = self.db.execute(query).all()
        return {row.rating: row.count for row in results}

    def get_feedback_type_distribution(self, session_id: str | None = None) -> dict[str, int]:
        query = select(
            UserFeedback.feedback_type,
            func.count(UserFeedback.id).label("count")
        ).group_by(UserFeedback.feedback_type)
        if session_id:
            query = query.where(UserFeedback.session_id == session_id)
        results = self.db.execute(query).all()
        return {row.feedback_type: row.count for row in results}

    def clear_expired(self, ttl_days: int = 30) -> int:
        from sqlalchemy import delete
        cutoff = datetime.utcnow() - timedelta(days=ttl_days)
        result = self.db.execute(delete(UserFeedback).where(UserFeedback.created_at < cutoff))
        self.db.commit()
        return result.rowcount or 0
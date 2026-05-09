from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    trace_id: str = Field(
        description="Trace ID from the original query response",
        examples=["trace_abc123"]
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation tracking",
        examples=["chat_123:user_456"]
    )
    query: str | None = Field(
        default=None,
        description="Original query text (optional, for reference)",
        examples=["山东的电力交易规则是什么？"]
    )
    answer: str | None = Field(
        default=None,
        description="Original answer text (optional, for reference)",
        examples=["根据《山东省电力市场交易规则》..."]
    )
    rating: int = Field(
        ge=1,
        le=5,
        description="User rating from 1 to 5 stars"
    )
    feedback_type: str = Field(
        description="Type of feedback: helpful, wrong, irrelevant, incomplete",
        examples=["helpful", "wrong", "irrelevant", "incomplete"]
    )
    user_comment: str | None = Field(
        default=None,
        description="Optional user comment",
        examples=["答案很详细，但缺少具体条款编号"]
    )
    suggested_correction: str | None = Field(
        default=None,
        description="Optional suggested correction from user",
        examples=["应该参考第十五条第二款"]
    )


class FeedbackResponse(BaseModel):
    id: int
    trace_id: str
    session_id: str | None
    query: str | None
    answer: str | None
    rating: int
    feedback_type: str
    user_comment: str | None
    suggested_correction: str | None
    created_at: datetime


class FeedbackStatsResponse(BaseModel):
    total_feedback: int
    avg_rating: float
    rating_distribution: dict[int, int]
    feedback_type_distribution: dict[str, int]
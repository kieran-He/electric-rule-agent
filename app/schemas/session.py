from __future__ import annotations

from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        description="Unique session identifier. If not provided, one will be generated."
    )
    channel: str = Field(
        default="api",
        description="Channel identifier for the session"
    )


class ExampleQuestion(BaseModel):
    question_id: str | None = None
    question: str
    category: str | None = None


class SessionCreateResponse(BaseModel):
    session_id: str
    is_new: bool = Field(description="Whether this is a new session")
    example_questions: list[ExampleQuestion] = Field(
        default_factory=list,
        description="Example questions for new sessions"
    )


class TitleGenerateRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        description="Session ID to generate title for"
    )


class TitleGenerateResponse(BaseModel):
    session_id: str
    title: str = Field(description="Generated title for the session")
    generated: bool = Field(description="Whether title was successfully generated")
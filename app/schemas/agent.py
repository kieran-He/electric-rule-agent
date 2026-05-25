from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.answer import CitationItem


class BenchmarkQuestion(BaseModel):
    question_id: str | None = None
    question: str
    category: str | None = None


class BenchmarkResponse(BaseModel):
    questions: List[BenchmarkQuestion]


class TitleRequest(BaseModel):
    session_id: str = Field(min_length=1, description="Session identifier")


class TitleResponse(BaseModel):
    session_id: str
    title: str
    generated: bool


class AgentRequest(BaseModel):
    query: str = Field(min_length=1, description="User query text")
    session_id: str = Field(min_length=1, description="Session identifier")
    province_codes: List[str] = Field(
        default_factory=lambda: ["SN"],
        description="Province codes to search"
    )
    history: List[str] = Field(default_factory=list, description="Conversation history")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class AgentResponse(BaseModel):
    answer: str = Field(description="Generated answer")
    intent: str = Field(description="Detected intent type")
    tool_calls: List[str] = Field(default_factory=list, description="Tools used")
    citations: List[CitationItem] = Field(default_factory=list, description="Citations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    confidence: float = Field(default=0.0, description="Answer confidence")
    trace_id: Optional[str] = Field(default=None, description="Trace ID for debugging")
    detected_provinces: Optional[str] = Field(default=None, description="Detected provinces")
    chart_paths: List[str] = Field(default_factory=list, description="Paths to generated chart images")
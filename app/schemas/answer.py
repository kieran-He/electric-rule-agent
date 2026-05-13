from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class CitationItem(BaseModel):
    doc_name: str
    status: str
    title_path: str
    article_no: str | None = None
    excerpt: str
    page_start: int | None = None
    page_end: int | None = None


class QueryAnswer(BaseModel):
    answer: str
    citations: list[CitationItem] = Field(default_factory=list)
    intent: str
    confidence: float = 0.0
    used_documents: list[str] = Field(default_factory=list)
    trace_id: str
    flow: str | None = None
    warnings: list[str] = Field(default_factory=list)
    detected_provinces: str | None = None
    verification: Optional[dict[str, Any]] = None


QueryResponse = QueryAnswer

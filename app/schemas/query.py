from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    province_codes: list[str] = Field(default_factory=lambda: ["SN"])
    mode: Literal["province_only", "province_plus_global"] = "province_only"
    top_k: int = 8
    need_citation: bool = True


class TraceResponse(BaseModel):
    trace_id: str
    raw_query: str
    rewritten_query: str | None = None
    intent: str | None = None
    filters: dict = Field(default_factory=dict)
    retrieved_doc_ids: list[int] = Field(default_factory=list)
    rerank_scores: list[float] = Field(default_factory=list)
    used_clause_ids: list[int] = Field(default_factory=list)
    final_doc_ids: list[int] = Field(default_factory=list)
    latency_ms: int = 0

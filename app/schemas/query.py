from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class QueryMode(str, Enum):
    auto = "auto"
    single_province = "single_province"
    province_plus_global = "province_plus_global"
    multi_province_compare = "multi_province_compare"


class KBScope(str, Enum):
    province = "province"
    global_scope = "global"


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="User query text",
        examples=["2026年陕西电力市场中长期交易流程是什么？"]
    )
    session_id: str = Field(
        min_length=1,
        description="Unique session identifier for conversation tracking",
        examples=["chat_123:user_456"]
    )
    province_codes: list[str] = Field(
        default_factory=lambda: ["SN"],
        description="Province codes to search (e.g., SN for Shaanxi)",
        examples=[["SN"], ["SN", "GD"]]
    )
    mode: Literal["province_only", "province_plus_global"] = Field(
        default="province_only",
        description="Search mode: province_only or province_plus_global"
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of results to retrieve"
    )
    need_citation: bool = Field(
        default=True,
        description="Whether to include citations in response"
    )


QueryRequestExample = QueryRequest(
    query="2026年陕西电力市场中长期交易流程是什么？",
    session_id="example_session_001",
    province_codes=["SN"]
)


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
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    retrieval_latency_ms: int | None = None
    llm_latency_ms: int | None = None
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None
    retrieved_doc_texts: list[str] = Field(default_factory=list)
    answer_text: str | None = None

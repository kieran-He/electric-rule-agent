from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

from app.schemas.answer import CitationItem


class AgentRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "陕西省电力市场化交易流程是什么？",
                    "session_id": "session_001",
                    "province_codes": ["SN"],
                    "show_chunks": True,
                }
            ]
        }
    )
    
    query: str = Field(
        min_length=1,
        description="User query text",
        examples=["陕西省电力市场化交易流程是什么？"]
    )
    session_id: str = Field(
        min_length=1,
        description="Session identifier",
        examples=["session_001"]
    )
    province_codes: List[str] = Field(
        default_factory=lambda: ["SN"],
        description="Province codes to search",
        examples=[["SN"], ["SN", "SX"]]
    )
    history: List[str] = Field(
        default_factory=list,
        description="Conversation history",
        examples=[["Q:上一个问题", "A:上一个回答"]]
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context",
        examples=[{"need_citation": True}]
    )
    show_chunks: bool = Field(
        default=True,
        description="Whether to show chunk references in answer",
        examples=[True]
    )


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
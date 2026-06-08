from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


DocumentStatus = Literal["formal", "trial", "notice", "draft"]


class RawPage(BaseModel):
    page_number: int
    text: str


class RuleTagExtraction(BaseModel):
    market_type: str | None = None
    entity_type: str | None = None
    trade_cycle: str | None = None
    trade_mode: str | None = None
    time_granularity: str | None = None
    action_type: str | None = None
    penalty_related: bool = False
    green_power_related: bool = False
    retail_related: bool = False
    storage_related: bool = False
    vpp_related: bool = False


class DocumentMetadata(BaseModel):
    province_code: str = "SN"
    doc_name: str
    doc_type: str = "unknown"
    status: DocumentStatus = "formal"
    issue_date: date | None = None
    effective_date: date | None = None
    source_file: str
    file_hash: str
    issuer: str | None = None


class ClauseChunk(BaseModel):
    doc_name: str
    source_file: str
    origin_doc_id: str | None = None
    province_code: str = "SN"
    doc_type: str = "unknown"
    doc_status: DocumentStatus = "formal"
    doc_issuer: str | None = None
    chapter_no: str | None = None
    chapter_title: str | None = None
    section_no: str | None = None
    section_title: str | None = None
    article_no: str | None = None
    item_no: str | None = None
    title_path: str
    clause_text: str
    clause_summary: str | None = None
    page_start: int = 1
    page_end: int = 1
    token_count: int
    rule_tags: RuleTagExtraction = Field(default_factory=RuleTagExtraction)


class ProcessingStats(BaseModel):
    total_pages: int = 0
    total_chapters: int = 0
    total_sections: int = 0
    total_articles: int = 0
    total_clauses: int = 0
    field_coverage: dict[str, int] = Field(default_factory=dict)


class ProcessedDocument(BaseModel):
    metadata: DocumentMetadata
    cleaned_text: str
    clauses: list[ClauseChunk]
    stats: ProcessingStats = Field(default_factory=ProcessingStats)
    ocr_used: bool = False
    processing_flags: list[str] = Field(default_factory=list)
    processed_at: datetime = Field(default_factory=datetime.utcnow)
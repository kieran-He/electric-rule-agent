from enum import Enum
from typing import List, Optional

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
    query: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    mode: QueryMode = QueryMode.auto
    province_codes: Optional[List[str]] = None
    top_k: Optional[int] = None


class Citation(BaseModel):
    province_code: Optional[str] = None
    source_name: str
    doc_id: str
    snippet: str
    policy_level: Optional[str] = None
    effective_date: Optional[str] = None


class QueryResponse(BaseModel):
    mode: QueryMode
    province_code: Optional[str] = None
    needs_confirmation: bool = False
    confirmation_question: Optional[str] = None
    conclusion: str
    provincial_evidence: List[Citation] = Field(default_factory=list)
    global_evidence: List[Citation] = Field(default_factory=list)
    differences: Optional[str] = None
    follow_up: str


class IngestRequest(BaseModel):
    docs_path: Optional[str] = None
    docs_root: Optional[str] = None
    kb_scope: KBScope = KBScope.province
    province_code: Optional[str] = None
    rebuild: bool = False
    dedupe: bool = True
    enable_ocr: Optional[bool] = None
    cleaning_profile: str = "robust"
    chunk_size: int = 800
    chunk_overlap: int = 120


class IngestResponse(BaseModel):
    success: bool
    files_processed: int
    chunks_created: int
    kb_scope: KBScope
    province_code: Optional[str]
    resolved_docs_path: str
    files_new: int = 0
    files_updated: int = 0
    files_skipped: int = 0
    ocr_pages_processed: int = 0
    message: str


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
    glm_ready: bool
    message: str

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = Field(min_length=1)
    province_code: str = "SN"
    rebuild_index: bool = False


class IngestResponse(BaseModel):
    success: bool
    imported_documents: int
    imported_clauses: int
    skipped_documents: int = 0
    message: str

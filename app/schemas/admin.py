from __future__ import annotations

from pydantic import BaseModel


class DocumentAdminItem(BaseModel):
    id: int
    doc_name: str
    doc_type: str
    status: str
    province_code: str
    version_name: str | None = None
    is_current: bool


class RebuildIndexResponse(BaseModel):
    success: bool
    message: str

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    detail: dict = Field(default_factory=dict)

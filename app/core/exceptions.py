from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppError(Exception):
    code: str
    message: str
    status_code: int = 400
    detail: dict[str, Any] = field(default_factory=dict)


class NotFoundEvidenceError(AppError):
    def __init__(self, message: str = "未找到相关内容", detail: dict[str, Any] | None = None):
        super().__init__("NO_CONTENT", message, 404, detail or {})


class DraftOnlyError(AppError):
    def __init__(self, detail: dict[str, Any] | None = None):
        super().__init__("DRAFT_ONLY", "仅找到征求意见稿，非正式执行文件", 200, detail or {})


class DatabaseUnavailableError(AppError):
    def __init__(self, message: str = "数据库连接失败"):
        super().__init__("DB_UNAVAILABLE", message, 503)


class VectorIndexUnavailableError(AppError):
    def __init__(self, message: str = "向量索引不可用"):
        super().__init__("VECTOR_UNAVAILABLE", message, 503)


class InvalidLLMResponseError(AppError):
    def __init__(self, message: str = "LLM 输出非法 JSON"):
        super().__init__("INVALID_LLM_JSON", message, 502)

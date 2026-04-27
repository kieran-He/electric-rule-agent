from __future__ import annotations

import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

from app.config import settings
from app.core.logging_context import LoggingContextFilter


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if hasattr(record, "trace_id") and record.trace_id:
            log_obj["trace_id"] = record.trace_id
        if hasattr(record, "session_id") and record.session_id:
            log_obj["session_id"] = record.session_id
        if hasattr(record, "request_id") and record.request_id:
            log_obj["request_id"] = record.request_id
        if hasattr(record, "user_id") and record.user_id:
            log_obj["user_id"] = record.user_id
        
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj, ensure_ascii=False)


def configure_logging() -> None:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(settings.log_level.upper())

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        settings.log_file,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    structured_file = settings.log_file.replace(".log", "_structured.json")
    structured_handler = RotatingFileHandler(
        structured_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    structured_handler.setFormatter(StructuredFormatter())
    structured_handler.addFilter(LoggingContextFilter())
    root.addHandler(structured_handler)

    if settings.feishu_alert_enabled and settings.feishu_webhook_url:
        from app.core.feishu_alert import create_feishu_handler
        feishu_handler = create_feishu_handler(settings.feishu_webhook_url)
        if feishu_handler:
            root.addHandler(feishu_handler)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)

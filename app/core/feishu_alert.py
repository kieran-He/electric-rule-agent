from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Optional

from app.core.logging_context import get_trace_id, get_session_id, get_request_id, get_user_id


class FeishuAlertHandler(logging.Handler):
    def __init__(
        self,
        webhook_url: str,
        level: int = logging.ERROR,
        timeout: int = 3,
    ):
        super().__init__(level)
        self.webhook_url = webhook_url
        self.timeout = timeout
    
    def emit(self, record: logging.LogRecord) -> None:
        if not self.webhook_url:
            return
        
        try:
            trace_id = getattr(record, 'trace_id', None) or get_trace_id()
            session_id = getattr(record, 'session_id', None) or get_session_id()
            request_id = getattr(record, 'request_id', None) or get_request_id()
            user_id = getattr(record, 'user_id', None) or get_user_id()
            
            context_parts = []
            if trace_id:
                context_parts.append(f"trace_id: {trace_id}")
            if session_id:
                context_parts.append(f"session_id: {session_id}")
            if request_id:
                context_parts.append(f"request_id: {request_id}")
            if user_id:
                context_parts.append(f"user_id: {user_id}")
            
            context_str = "\n".join(context_parts) if context_parts else ""
            
            message = f"""⚠️ [{record.levelname}] {record.name}
{record.getMessage()}
{context_str}""".strip()
            
            payload = {
                "msg_type": "text",
                "content": {
                    "text": message
                }
            }
            
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                response.read()
                
        except Exception:
            self.handleError(record)


def create_feishu_handler(webhook_url: Optional[str]) -> Optional[FeishuAlertHandler]:
    if not webhook_url:
        return None
    
    handler = FeishuAlertHandler(webhook_url)
    from app.core.logging_context import LoggingContextFilter
    handler.addFilter(LoggingContextFilter())
    return handler
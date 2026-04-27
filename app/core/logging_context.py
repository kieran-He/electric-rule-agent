from __future__ import annotations

import contextvars
from typing import Optional

_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("trace_id", default=None)
_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("session_id", default=None)
_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("user_id", default=None)


def set_trace_id(trace_id: str):
    _trace_id.set(trace_id)


def set_session_id(session_id: str):
    _session_id.set(session_id)


def set_request_id(request_id: str):
    _request_id.set(request_id)


def set_user_id(user_id: str):
    _user_id.set(user_id)


def get_trace_id() -> Optional[str]:
    return _trace_id.get()


def get_session_id() -> Optional[str]:
    return _session_id.get()


def get_request_id() -> Optional[str]:
    return _request_id.get()


def get_user_id() -> Optional[str]:
    return _user_id.get()


class LoggingContextFilter:
    def filter(self, record) -> bool:
        record.trace_id = get_trace_id()
        record.session_id = get_session_id()
        record.request_id = get_request_id()
        record.user_id = get_user_id()
        return True
from __future__ import annotations

import logging

from app.core.logging_context import (
    set_trace_id,
    set_session_id,
    set_request_id,
    set_user_id,
    get_trace_id,
    get_session_id,
    get_request_id,
    get_user_id,
    LoggingContextFilter,
)


def test_set_and_get_trace_id():
    set_trace_id("trace_abc123")
    assert get_trace_id() == "trace_abc123"


def test_set_and_get_session_id():
    set_session_id("session_def456")
    assert get_session_id() == "session_def456"


def test_set_and_get_request_id():
    set_request_id("request_ghi789")
    assert get_request_id() == "request_ghi789"


def test_set_and_get_user_id():
    set_user_id("user_jkl012")
    assert get_user_id() == "user_jkl012"


def test_get_default_none():
    import contextvars
    contextvars.ContextVar("trace_id_test", default=None).set(None)
    
    temp_trace_id = contextvars.ContextVar("temp_trace", default=None)
    assert temp_trace_id.get() is None


def test_logging_context_filter():
    set_trace_id("filter_trace_001")
    set_session_id("filter_session_001")
    set_request_id("filter_request_001")
    set_user_id("filter_user_001")
    
    filter_obj = LoggingContextFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    
    result = filter_obj.filter(record)
    assert result is True
    assert record.trace_id == "filter_trace_001"
    assert record.session_id == "filter_session_001"
    assert record.request_id == "filter_request_001"
    assert record.user_id == "filter_user_001"


def test_context_isolation():
    import threading
    
    set_trace_id("main_trace")
    
    results = {}
    
    def thread_func():
        set_trace_id("thread_trace")
        results["thread"] = get_trace_id()
    
    thread = threading.Thread(target=thread_func)
    thread.start()
    thread.join()
    
    assert results["thread"] == "thread_trace"
    assert get_trace_id() == "main_trace"


def test_multiple_set_calls():
    set_trace_id("first_trace")
    assert get_trace_id() == "first_trace"
    
    set_trace_id("second_trace")
    assert get_trace_id() == "second_trace"
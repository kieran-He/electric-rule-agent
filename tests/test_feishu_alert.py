from __future__ import annotations

import logging

from app.core.feishu_alert import FeishuAlertHandler, create_feishu_handler
from app.core.logging_context import set_trace_id, set_session_id, set_request_id, set_user_id


def test_feishu_alert_handler_init():
    handler = FeishuAlertHandler("https://example.com/webhook")
    assert handler.webhook_url == "https://example.com/webhook"
    assert handler.timeout == 3
    assert handler.level == logging.ERROR


def test_feishu_alert_handler_custom_timeout():
    handler = FeishuAlertHandler("https://example.com/webhook", timeout=10)
    assert handler.timeout == 10


def test_feishu_alert_handler_custom_level():
    handler = FeishuAlertHandler("https://example.com/webhook", level=logging.WARNING)
    assert handler.level == logging.WARNING


def test_feishu_alert_handler_emit_no_webhook():
    handler = FeishuAlertHandler("")
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg="Test error",
        args=(),
        exc_info=None,
    )
    handler.emit(record)


def test_create_feishu_handler_with_url():
    handler = create_feishu_handler("https://example.com/webhook")
    assert handler is not None
    assert handler.webhook_url == "https://example.com/webhook"


def test_create_feishu_handler_no_url():
    handler = create_feishu_handler("")
    assert handler is None


def test_create_feishu_handler_none_url():
    handler = create_feishu_handler(None)
    assert handler is None


def test_feishu_alert_handler_with_context():
    set_trace_id("trace_test_001")
    set_session_id("session_test_001")
    set_request_id("request_test_001")
    set_user_id("user_test_001")
    
    handler = FeishuAlertHandler("https://example.com/webhook")
    record = logging.LogRecord(
        name="test.logger",
        level=logging.ERROR,
        pathname="test.py",
        lineno=42,
        msg="Test error message",
        args=(),
        exc_info=None,
    )
    handler.emit(record)


def test_feishu_alert_handler_record_attributes():
    handler = FeishuAlertHandler("https://example.com/webhook")
    record = logging.LogRecord(
        name="test.logger",
        level=logging.ERROR,
        pathname="test.py",
        lineno=42,
        msg="Test error",
        args=(),
        exc_info=None,
    )
    record.trace_id = "record_trace_001"
    record.session_id = "record_session_001"
    
    handler.emit(record)
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.services.feishu_agent_service import FeishuAgentService
from app.schemas.agent import AgentResponse


class FakeSession:
    def __init__(self):
        self._committed = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    
    def commit(self):
        self._committed = True
    
    def query(self, model):
        return MagicMock(filter=lambda *args: MagicMock(first=lambda: None))
    
    def add(self, obj):
        pass
    
    def delete(self):
        pass


def build_feishu_agent_service():
    settings = SimpleNamespace(
        province_defaults=["SN"],
        coreference_resolution_enabled=False,
    )
    
    session_factory = lambda: FakeSession()
    client = MagicMock()
    
    service = FeishuAgentService(
        settings=settings,
        session_factory=session_factory,
        client=client,
    )
    
    return service


def test_feishu_agent_service_get_session_id():
    service = build_feishu_agent_service()
    
    session_id = service._get_session_id("user123")
    assert session_id == "feishu:user123"


def test_feishu_agent_service_extract_message_text():
    service = build_feishu_agent_service()
    
    message = MagicMock()
    message.content = '{"text": "电力交易流程"}'
    
    text = service._extract_message_text(message)
    assert text == "电力交易流程"


def test_feishu_agent_service_remove_mention():
    service = build_feishu_agent_service()
    
    text = "<at user_id=\"ou_xxx\">@机器人</at>电力交易流程"
    cleaned = service._remove_mention(text)
    assert cleaned == "电力交易流程"


def test_feishu_agent_service_format_reply():
    service = build_feishu_agent_service()
    
    response = AgentResponse(
        answer="电力市场交易规则规定...",
        intent="policy_qa",
        tool_calls=["rag"],
        confidence=0.8,
    )
    
    reply = service._format_reply(response)
    assert reply == "电力市场交易规则规定..."


def test_feishu_agent_service_process_with_agent():
    service = build_feishu_agent_service()
    
    mock_agent = MagicMock()
    mock_agent.chat.return_value = AgentResponse(
        answer="电力市场交易规则规定...",
        intent="policy_qa",
        tool_calls=["rag"],
        confidence=0.8,
    )
    
    with patch.object(service.conversation_service, 'get_history', return_value=[]):
        with patch('app.services.feishu_agent_service.agent_singleton.get_agent', return_value=mock_agent):
            response = service._process_with_agent(
                text="电力交易流程",
                session_id="feishu:test",
                trace_id="trace_test",
            )
            
            assert response.intent == "policy_qa"
            assert response.tool_calls == ["rag"]
            mock_agent.chat.assert_called_once()


def test_feishu_agent_service_deduplication():
    service = build_feishu_agent_service()
    
    mock_query_result = MagicMock()
    mock_query_result.first.return_value = None
    mock_filter_result = MagicMock()
    mock_filter_result.filter.return_value = mock_query_result
    
    with patch.object(FakeSession, 'query', return_value=mock_filter_result):
        first_check = service._check_and_mark("event123")
        assert first_check == False
    
    mock_existing = MagicMock()
    mock_query_result2 = MagicMock()
    mock_query_result2.first.return_value = mock_existing
    mock_filter_result2 = MagicMock()
    mock_filter_result2.filter.return_value = mock_query_result2
    
    with patch.object(FakeSession, 'query', return_value=mock_filter_result2):
        second_check = service._check_and_mark("event123")
        assert second_check == True
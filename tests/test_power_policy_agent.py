from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.agent.intent_router import IntentRouter, IntentType
from app.agent.power_policy_agent import PowerPolicyAgent
from app.agent.tools.rag_tool import RAGTool
from app.agent.tools.web_search_tool import WebSearchTool
from app.agent.tools.general_chat_tool import GeneralChatTool
from app.schemas.agent import AgentRequest, AgentResponse
from app.schemas.answer import QueryAnswer


class FakeOrchestrator:
    def __init__(self):
        self.db = None
    
    def run(self, req, history=None, trace_service=None, db=None, rewrite_result=None):
        return QueryAnswer(
            answer="电力市场交易规则规定...",
            citations=[],
            intent="policy_qa",
            confidence=0.8,
            used_documents=["陕西省电力市场交易规则"],
            trace_id="trace_test",
            flow=None,
            warnings=[],
            detected_provinces="SN",
        )


class FakeLLMWrapper:
    api_key = None
    
    def invoke(self, prompt, system=None):
        return ("通用回答", 50, 30)


def build_agent():
    orchestrator = FakeOrchestrator()
    llm_wrapper = FakeLLMWrapper()
    settings = SimpleNamespace(
        web_search_include_gov=True,
    )
    
    return PowerPolicyAgent(
        orchestrator=orchestrator,
        llm_wrapper=llm_wrapper,
        settings=settings,
        web_search_client=None,
    )


def test_intent_router_routes_rag():
    router = IntentRouter()
    
    orchestrator = FakeOrchestrator()
    llm_wrapper = FakeLLMWrapper()
    settings = SimpleNamespace()
    
    tools = [
        RAGTool(orchestrator),
        GeneralChatTool(llm_wrapper),
    ]
    
    selected = router.route("电力交易流程是什么？", tools)
    assert selected.name == "rag"
    
    intent = router.detect_intent("电力交易流程", selected)
    assert intent == IntentType.POLICY_QA


def test_intent_router_defaults_to_rag():
    router = IntentRouter()
    
    orchestrator = FakeOrchestrator()
    llm_wrapper = FakeLLMWrapper()
    
    tools = [
        RAGTool(orchestrator),
        GeneralChatTool(llm_wrapper),
    ]
    
    selected = router.route("非电力问题", tools)
    assert selected.name == "rag"
    
    intent = router.detect_intent("非电力问题", selected)
    assert intent == IntentType.POLICY_QA


def test_intent_router_web_search_priority():
    router = IntentRouter()
    
    orchestrator = FakeOrchestrator()
    llm_wrapper = FakeLLMWrapper()
    web_client = MagicMock()
    web_client.is_available.return_value = True
    settings = SimpleNamespace(web_search_include_gov=True)
    
    tools = [
        RAGTool(orchestrator),
        WebSearchTool(web_client, llm_wrapper, settings),
        GeneralChatTool(llm_wrapper),
    ]
    
    selected = router.route("今天的最新新闻头条", tools)
    assert selected.name == "web_search"
    
    intent = router.detect_intent("今天最新新闻", selected)
    assert intent == IntentType.WEB_SEARCH


def test_intent_router_rag_has_priority_over_web_search():
    router = IntentRouter()
    
    orchestrator = FakeOrchestrator()
    llm_wrapper = FakeLLMWrapper()
    web_client = MagicMock()
    web_client.is_available.return_value = True
    settings = SimpleNamespace(web_search_include_gov=True)
    
    tools = [
        RAGTool(orchestrator),
        WebSearchTool(web_client, llm_wrapper, settings),
        GeneralChatTool(llm_wrapper),
    ]
    
    selected = router.route("最新的电力政策新闻", tools)
    assert selected.name == "rag"
    
    intent = router.detect_intent("最新电力政策", selected)
    assert intent == IntentType.POLICY_QA


def test_power_policy_agent_chat():
    agent = build_agent()
    
    request = AgentRequest(
        query="电力交易流程",
        session_id="test_session",
        province_codes=["SN"],
        history=["Q: 之前的问题", "A: 之前的回答"],
    )
    
    response = agent.chat(request)
    
    assert isinstance(response, AgentResponse)
    assert response.intent == "policy_qa"
    assert response.tool_calls == ["rag"]
    assert response.confidence == 0.8
    assert "电力市场交易规则" in response.answer


def test_power_policy_agent_stats():
    agent = build_agent()
    
    stats = agent.get_stats()
    
    assert "tools" in stats
    assert "rag" in stats["tools"]
    assert "general_chat" in stats["tools"]
    assert stats["router"] == "keyword_match"


def test_power_policy_agent_general_chat_fallback():
    orchestrator = FakeOrchestrator()
    llm_wrapper = MagicMock()
    llm_wrapper.invoke.return_value = ("通用回答", 50, 30)
    llm_wrapper.api_key = None
    settings = SimpleNamespace()
    
    agent = PowerPolicyAgent(
        orchestrator=orchestrator,
        llm_wrapper=llm_wrapper,
        settings=settings,
        web_search_client=None,
        use_react=False,
    )
    
    request = AgentRequest(
        query="电力交易流程",
        session_id="test_session",
        province_codes=["SN"],
        history=[],
    )
    
    response = agent.chat(request)
    
    assert response.tool_calls == ["rag"]
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.agent.tools.base import BaseTool, ToolResult
from app.agent.tools.rag_tool import RAGTool
from app.agent.tools.web_search_tool import WebSearchTool
from app.agent.tools.general_chat_tool import GeneralChatTool
from app.schemas.answer import QueryAnswer, CitationItem


class FakeOrchestrator:
    def run(self, req, history=None, trace_service=None, db=None, rewrite_result=None):
        return QueryAnswer(
            answer="电力市场交易规则规定...",
            citations=[CitationItem(
                doc_name="陕西省电力市场交易规则",
                status="formal",
                title_path="第一章 总则",
                excerpt="交易流程...",
            )],
            intent="policy_qa",
            confidence=0.8,
            used_documents=["陕西省电力市场交易规则"],
            trace_id="trace_test",
            flow=None,
            warnings=[],
            detected_provinces="SN",
        )


def test_rag_tool_is_applicable():
    orchestrator = FakeOrchestrator()
    tool = RAGTool(orchestrator)
    
    assert tool.is_applicable("电力市场交易流程是什么？")
    assert tool.is_applicable("陕西电网的规则有哪些？")
    assert tool.is_applicable("电价结算方式")
    assert tool.is_applicable("新能源发电政策")
    
    assert not tool.is_applicable("今天天气怎么样？")
    assert not tool.is_applicable("推荐一本好书")


def test_rag_tool_execute():
    orchestrator = FakeOrchestrator()
    tool = RAGTool(orchestrator)
    
    context = {
        "session_id": "test_session",
        "province_codes": ["SN"],
        "history": [],
        "rewrite_result": None,
    }
    
    result = tool.execute("电力交易流程", context)
    
    assert result.success
    assert "电力市场交易规则" in result.output
    assert result.tool_name == "rag"
    assert result.confidence == 0.8
    assert len(result.citations) == 1


def test_web_search_tool_is_applicable():
    llm_wrapper = MagicMock()
    llm_wrapper.invoke.return_value = ("搜索结果答案", 100, 50)
    
    web_client = MagicMock()
    web_client.is_available.return_value = True
    web_client.search.return_value = [
        {"title": "最新新闻", "content": "新闻内容", "url": "http://example.com"}
    ]
    web_client.format_results_for_context.return_value = "1. 最新新闻\n   新闻内容"
    
    settings = SimpleNamespace(web_search_include_gov=True)
    tool = WebSearchTool(web_client, llm_wrapper, settings)
    
    assert tool.is_applicable("最新的电力政策新闻")
    assert tool.is_applicable("今天的电价变化")
    assert tool.is_applicable("近期电网动态")
    
    assert not tool.is_applicable("电力交易流程")


def test_web_search_tool_execute():
    llm_wrapper = MagicMock()
    llm_wrapper.invoke.return_value = ("搜索结果答案", 100, 50)
    
    web_client = MagicMock()
    web_client.is_available.return_value = True
    web_client.search.return_value = [
        {"title": "最新新闻", "content": "新闻内容", "url": "http://example.com"}
    ]
    web_client.format_results_for_context.return_value = "1. 最新新闻\n   新闻内容"
    
    settings = SimpleNamespace(web_search_include_gov=True)
    tool = WebSearchTool(web_client, llm_wrapper, settings)
    
    context = {"session_id": "test", "province_codes": ["SN"], "missing_provinces": None, "rewrite_result": None}
    result = tool.execute("最新电力政策", context)
    
    assert result.success
    assert "网络搜索" in result.output
    assert result.tool_name == "web_search"


def test_web_search_tool_not_available():
    llm_wrapper = MagicMock()
    web_client = MagicMock()
    web_client.is_available.return_value = False
    
    settings = SimpleNamespace()
    tool = WebSearchTool(web_client, llm_wrapper, settings)
    
    context = {"session_id": "test"}
    result = tool.execute("最新消息", context)
    
    assert not result.success
    assert "暂不可用" in result.output


def test_general_chat_tool_is_applicable():
    llm_wrapper = MagicMock()
    tool = GeneralChatTool(llm_wrapper)
    
    assert not tool.is_applicable("电力交易流程")
    assert not tool.is_applicable("今天天气")
    
    assert tool.keywords == []


def test_general_chat_tool_execute():
    llm_wrapper = MagicMock()
    llm_wrapper.invoke.return_value = ("我是一个电力政策问答助手，主要回答电力相关问题。", 50, 30)
    
    tool = GeneralChatTool(llm_wrapper)
    
    context = {"history": ["Q: 电价是多少？", "A: 约0.5元/度"]}
    result = tool.execute("你叫什么名字？", context)
    
    assert result.success
    assert result.tool_name == "general_chat"


def test_general_chat_tool_handles_error():
    llm_wrapper = MagicMock()
    llm_wrapper.invoke.side_effect = Exception("LLM error")
    
    tool = GeneralChatTool(llm_wrapper)
    
    context = {}
    result = tool.execute("test query", context)
    
    assert not result.success
    assert "无法回答" in result.output
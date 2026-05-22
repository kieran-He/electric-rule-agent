import logging
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from app.agent.graph.electricity_agent_graph import ElectricityAgentGraph
from app.agent.graph.nodes.react_agent_node import _fill_and_renumber_citations, react_agent_node
from app.agent.graph.state import create_initial_state
from app.schemas.agent import AgentRequest


def _chunks(count: int):
    return [
        {
            "source": f"doc-{i}",
            "title_path": f"title-{i}",
            "content": f"content-{i}",
        }
        for i in range(1, count + 1)
    ]


def test_fill_and_renumber_citations_keeps_appearance_order():
    answer = "A [chunk-3] then B [chunk-1] then C [chunk-3]."
    chunks = _chunks(4)

    processed, ordered = _fill_and_renumber_citations(answer, chunks)

    assert "[doc-3](#chunk-1)" in processed
    assert "[doc-1](#chunk-2)" in processed
    assert len(ordered) == 2
    assert ordered[0]["source"] == "doc-3"
    assert ordered[1]["source"] == "doc-1"


def test_fill_and_renumber_citations_mixed_formats():
    answer = (
        "x [引用](#chunk-2) y [xxx](#chunk-4) z [chunk-1] and #chunk-3"
    )
    chunks = _chunks(4)

    processed, ordered = _fill_and_renumber_citations(answer, chunks)

    assert [c["source"] for c in ordered] == ["doc-2", "doc-4", "doc-1", "doc-3"]
    assert "[doc-2](#chunk-1)" in processed
    assert "[doc-4](#chunk-2)" in processed
    assert "[doc-1](#chunk-3)" in processed
    assert "[doc-3](#chunk-4)" in processed


def test_fill_and_renumber_citations_invalid_indices_warning(caplog):
    answer = "bad #chunk-0 and #chunk-999 and #chunk-abc"
    chunks = _chunks(4)

    with caplog.at_level(logging.WARNING):
        processed, ordered = _fill_and_renumber_citations(answer, chunks)

    assert processed == answer
    assert ordered == []
    assert "no valid indices" in caplog.text


def test_fill_and_renumber_citations_fallback_by_source_name():
    answer = "根据《河南电力交易规则》要求，市场主体应满足准入条件。"
    chunks = [
        {"source": "陕西规则", "title_path": "", "content": "无关内容"},
        {"source": "河南电力交易规则", "title_path": "", "content": "发电企业、售电公司可参与"},
    ]

    processed, ordered = _fill_and_renumber_citations(answer, chunks)

    assert processed == answer
    assert len(ordered) == 1
    assert ordered[0]["source"] == "河南电力交易规则"


def test_fill_and_renumber_citations_fallback_by_content_phrase():
    answer = "交易主体应具有独立法人资格并独立承担民事责任。"
    chunks = [
        {"source": "doc-1", "title_path": "", "content": "经营主体应具有独立法人资格并独立承担民事责任。"},
        {"source": "doc-2", "title_path": "", "content": "另一个不相关片段"},
    ]

    processed, ordered = _fill_and_renumber_citations(answer, chunks)

    assert processed == answer
    assert len(ordered) == 1
    assert ordered[0]["source"] == "doc-1"


def test_chat_returns_citations_from_reordered_policy_chunks():
    state = create_initial_state(query="test", provinces=["SN"], max_iterations=5)
    state["policy_chunks"] = [
        {"source": "doc-A", "content": "A content"},
        {"source": "doc-B", "content": "B content"},
    ]

    mock_graph = MagicMock()
    mock_graph.settings = MagicMock()
    mock_graph.settings.tools_enabled_list = None
    mock_graph.llm_wrapper = MagicMock()
    mock_graph.llm_wrapper.invoke_with_tools.return_value = AIMessage(
        content="answer [chunk-2] then [chunk-1]"
    )

    with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
        react_result = react_agent_node(state)

    graph = ElectricityAgentGraph.__new__(ElectricityAgentGraph)
    graph.run = MagicMock(return_value={
        "answer": react_result["answer"],
        "policy_chunks": react_result["policy_chunks"],
        "intent": "",
        "tool_calls": [],
        "metadata": {},
        "confidence": 0.9,
        "chart_paths": [],
    })

    request = AgentRequest(query="q", session_id="s1", province_codes=["SN"])
    resp = graph.chat(request)

    assert len(resp.citations) == 2
    assert resp.citations[0].doc_name == "doc-B"
    assert resp.citations[1].doc_name == "doc-A"

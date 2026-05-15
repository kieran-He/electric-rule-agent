import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from app.agent.graph.nodes.react_agent_node import react_agent_node, _build_messages
from app.agent.graph.state import ElectricityAgentState, create_initial_state


class TestReactAgentNode:
    def test_react_agent_node_no_graph_instance(self):
        state = create_initial_state(
            query="电力交易规则",
            provinces=["SN"],
            max_iterations=5,
        )
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=None):
            result = react_agent_node(state)
            
            assert result["done"] == True
            assert "错误" in result["answer"]
    
    def test_react_agent_node_with_tool_calls(self):
        state = create_initial_state(
            query="电力交易规则",
            provinces=["SN"],
            max_iterations=5,
        )
        
        mock_graph = MagicMock()
        mock_llm = MagicMock()
        
        mock_response = AIMessage(content="", tool_calls=[
            {"name": "retrieve_policy", "args": {"query": "电力交易规则"}, "id": "call_1"}
        ])
        
        mock_llm.invoke_with_tools.return_value = mock_response
        mock_graph.llm_wrapper = mock_llm
        mock_graph.settings = MagicMock()
        mock_graph.settings.tools_enabled_list = None
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            result = react_agent_node(state)
            
            assert result["done"] == False
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["name"] == "retrieve_policy"
    
    def test_react_agent_node_final_answer(self):
        state = create_initial_state(
            query="电力交易规则",
            provinces=["SN"],
            max_iterations=5,
        )
        
        mock_graph = MagicMock()
        mock_llm = MagicMock()
        
        mock_response = AIMessage(content="根据检索结果，陕西省电力市场交易规则...")
        
        mock_llm.invoke_with_tools.return_value = mock_response
        mock_graph.llm_wrapper = mock_llm
        mock_graph.settings = MagicMock()
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            result = react_agent_node(state)
            
            assert result["done"] == True
            assert "交易规则" in result["answer"]
    
    def test_react_agent_iteration_increment(self):
        state = create_initial_state(
            query="电力交易规则",
            provinces=["SN"],
            max_iterations=5,
        )
        
        mock_graph = MagicMock()
        mock_llm = MagicMock()
        
        mock_response = AIMessage(content="", tool_calls=[
            {"name": "retrieve_policy", "args": {}, "id": "call_1"}
        ])
        
        mock_llm.invoke_with_tools.return_value = mock_response
        mock_graph.llm_wrapper = mock_llm
        mock_graph.settings = MagicMock()
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            result = react_agent_node(state)
            
            assert result["iteration_count"] == 1


class TestBuildMessages:
    def test_build_messages_with_history(self):
        state = create_initial_state(
            query="最新问题",
            provinces=["SN"],
            history=[{"role": "user", "content": "之前问题"}, {"role": "assistant", "content": "之前回答"}],
            max_iterations=5,
        )
        
        messages = _build_messages(state)
        
        assert len(messages) >= 3
    
    def test_build_messages_initial_query(self):
        state = create_initial_state(
            query="电力交易规则",
            provinces=["SN"],
            max_iterations=5,
        )
        
        messages = _build_messages(state)
        
        assert len(messages) >= 2
    
    def test_build_messages_with_tool_results(self):
        state = create_initial_state(
            query="电力交易规则",
            provinces=["SN"],
            max_iterations=1,
        )
        state["tool_results"] = [
            {"tool_name": "retrieve_policy", "output": "政策内容", "tool_call_id": "call_1"}
        ]
        state["last_tool_calls"] = [
            {"name": "retrieve_policy", "args": {}, "id": "call_1"}
        ]
        
        messages = _build_messages(state)
        
        assert len(messages) >= 3
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from app.agent.graph.state import create_initial_state
from app.agent.graph.nodes.react_agent_node import react_agent_node
from app.agent.graph.nodes.tool_executor_node import tool_executor_node
from app.agent.graph.electricity_agent_graph import _should_continue


class TestFullFlow:
    def test_full_flow_policy_query(self):
        state = create_initial_state(
            query="陕西省电力交易规则",
            provinces=["SN"],
            max_iterations=5,
        )
        
        mock_graph = MagicMock()
        mock_llm = MagicMock()
        
        mock_llm.invoke_with_tools.side_effect = [
            AIMessage(content="", tool_calls=[
                {"name": "retrieve_policy", "args": {"query": "电力交易规则"}, "id": "call_1"}
            ]),
            AIMessage(content="根据检索结果，陕西省电力市场交易规则主要内容包括..."),
        ]
        
        mock_graph.llm_wrapper = mock_llm
        mock_graph.settings = MagicMock()
        mock_graph.settings.tools_enabled_list = None
        
        from app.agent.graph.handlers.iteration_control import IterationController
        mock_graph._iteration_controller = IterationController(max_iterations=5, timeout_seconds=30)
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            result1 = react_agent_node(state)
            
            assert result1["done"] == False
            assert len(result1["tool_calls"]) == 1
            assert result1["tool_calls"][0]["name"] == "retrieve_policy"
            
            state.update(result1)
            
            result2 = tool_executor_node(state)
            
            assert len(result2["tool_results"]) == 1
            assert result2["tool_results"][0]["success"] == True
            
            state.update(result2)
            state["iteration_count"] = result1["iteration_count"]
            
            result3 = react_agent_node(state)
            
            assert result3["done"] == True
            assert len(result3["answer"]) > 0
    
    def test_full_flow_data_query(self):
        state = create_initial_state(
            query="陕西省近期负荷数据",
            provinces=["SN"],
            max_iterations=5,
        )
        
        mock_graph = MagicMock()
        mock_llm = MagicMock()
        
        mock_llm.invoke_with_tools.side_effect = [
            AIMessage(content="", tool_calls=[
                {"name": "fetch_electricity_data", "args": {"province": "SN", "metric": "load"}, "id": "call_1"}
            ]),
            AIMessage(content="根据数据查询结果，陕西省近期负荷数据显示..."),
        ]
        
        mock_graph.llm_wrapper = mock_llm
        mock_graph.settings = MagicMock()
        mock_graph.settings.tools_enabled_list = None
        
        from app.agent.graph.handlers.iteration_control import IterationController
        mock_graph._iteration_controller = IterationController(max_iterations=5, timeout_seconds=30)
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            result1 = react_agent_node(state)
            assert result1["tool_calls"][0]["name"] == "fetch_electricity_data"
            
            state.update(result1)
            result2 = tool_executor_node(state)
            assert result2["tool_results"][0]["success"] == True
            
            state.update(result2)
            state["iteration_count"] = result1["iteration_count"]
            
            result3 = react_agent_node(state)
            assert result3["done"] == True
    
    def test_full_flow_with_history(self):
        state = create_initial_state(
            query="继续上一个问题",
            provinces=["SN"],
            history=[
                {"role": "user", "content": "陕西省电力交易规则"},
                {"role": "assistant", "content": "陕西省电力市场交易规则主要内容包括..."},
            ],
            max_iterations=5,
        )
        
        mock_graph = MagicMock()
        mock_llm = MagicMock()
        mock_llm.invoke_with_tools.return_value = AIMessage(content="根据之前的讨论，继续分析...")
        
        mock_graph.llm_wrapper = mock_llm
        mock_graph.settings = MagicMock()
        mock_graph.settings.tools_enabled_list = None
        
        from app.agent.graph.handlers.iteration_control import IterationController
        mock_graph._iteration_controller = IterationController(max_iterations=5, timeout_seconds=30)
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            result = react_agent_node(state)
            
            assert result["done"] == True
            assert len(result["answer"]) > 0
    
    def test_routing_decision_with_tool_calls(self):
        state = create_initial_state(
            query="测试",
            provinces=["SN"],
            max_iterations=5,
        )
        state["tool_calls"] = [{"name": "retrieve_policy", "args": {}, "id": "call_1"}]
        state["iteration_count"] = 1
        
        mock_graph = MagicMock()
        from app.agent.graph.handlers.iteration_control import IterationController
        mock_graph._iteration_controller = IterationController(max_iterations=5, timeout_seconds=30)
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            route = _should_continue(state)
            assert route == "tools"
    
    def test_routing_decision_done(self):
        state = create_initial_state(
            query="测试",
            provinces=["SN"],
            max_iterations=5,
        )
        state["done"] = True
        
        route = _should_continue(state)
        assert route == "end"
    
    def test_routing_decision_max_iterations(self):
        state = create_initial_state(
            query="测试",
            provinces=["SN"],
            max_iterations=3,
        )
        state["iteration_count"] = 3
        
        route = _should_continue(state)
        assert route == "end"
    
    def test_thought_chain_recorded(self):
        state = create_initial_state(
            query="测试",
            provinces=["SN"],
            max_iterations=5,
        )
        
        mock_graph = MagicMock()
        mock_llm = MagicMock()
        
        mock_llm.invoke_with_tools.side_effect = [
            AIMessage(content="", tool_calls=[
                {"name": "retrieve_policy", "args": {}, "id": "call_1"}
            ]),
            AIMessage(content="最终答案"),
        ]
        
        mock_graph.llm_wrapper = mock_llm
        mock_graph.settings = MagicMock()
        mock_graph.settings.tools_enabled_list = None
        
        from app.agent.graph.handlers.iteration_control import IterationController
        mock_graph._iteration_controller = IterationController(max_iterations=5, timeout_seconds=30)
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            result1 = react_agent_node(state)
            thoughts = result1.get("thoughts", [])
            assert len(thoughts) > 0
            assert any(t.get("phase") == "thinking" for t in thoughts)
            assert any(t.get("phase") == "tool_call" for t in thoughts)
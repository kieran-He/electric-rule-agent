import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from app.agent.graph.state import create_initial_state
from app.agent.graph.nodes.react_agent_node import react_agent_node
from app.agent.graph.nodes.tool_executor_node import tool_executor_node


class TestMultiToolCombination:
    def test_policy_then_data_combination(self):
        state = create_initial_state(
            query="陕西省电力交易规则及负荷数据",
            provinces=["SN"],
            max_iterations=5,
        )
        
        mock_graph = MagicMock()
        mock_llm = MagicMock()
        
        mock_llm.invoke_with_tools.side_effect = [
            AIMessage(content="", tool_calls=[
                {"name": "retrieve_policy", "args": {"query": "电力交易规则"}, "id": "call_1"}
            ]),
            AIMessage(content="", tool_calls=[
                {"name": "fetch_electricity_data", "args": {"province": "SN", "metric": "load"}, "id": "call_2"}
            ]),
            AIMessage(content="综合政策规定和数据情况，陕西省电力市场..."),
        ]
        
        mock_graph.llm_wrapper = mock_llm
        mock_graph.settings = MagicMock()
        mock_graph.settings.tools_enabled_list = None
        
        from app.agent.graph.handlers.iteration_control import IterationController
        mock_graph._iteration_controller = IterationController(max_iterations=5, timeout_seconds=30)
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            result1 = react_agent_node(state)
            assert result1["tool_calls"][0]["name"] == "retrieve_policy"
            
            state.update(result1)
            result2 = tool_executor_node(state)
            assert len(state["policy_chunks"]) > 0
            
            state.update(result2)
            state["iteration_count"] = result1["iteration_count"]
            
            result3 = react_agent_node(state)
            assert result3["tool_calls"][0]["name"] == "fetch_electricity_data"
            
            state.update(result3)
            result4 = tool_executor_node(state)
            assert state["electricity_data"] is not None
            
            state.update(result4)
            state["iteration_count"] = result3["iteration_count"]
            
            result5 = react_agent_node(state)
            assert result5["done"] == True
    
    def test_data_then_analysis_combination(self):
        state = create_initial_state(
            query="分析陕西省负荷数据趋势",
            provinces=["SN"],
            max_iterations=5,
        )
        
        mock_graph = MagicMock()
        mock_llm = MagicMock()
        
        mock_llm.invoke_with_tools.side_effect = [
            AIMessage(content="", tool_calls=[
                {"name": "fetch_electricity_data", "args": {"province": "SN", "metric": "load"}, "id": "call_1"}
            ]),
            AIMessage(content="", tool_calls=[
                {"name": "analyze_statistics", "args": {"analysis_type": "trend"}, "id": "call_2"}
            ]),
            AIMessage(content="根据数据分析结果，负荷呈现上升趋势..."),
        ]
        
        mock_graph.llm_wrapper = mock_llm
        mock_graph.settings = MagicMock()
        mock_graph.settings.tools_enabled_list = None
        
        from app.agent.graph.handlers.iteration_control import IterationController
        mock_graph._iteration_controller = IterationController(max_iterations=5, timeout_seconds=30)
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            result1 = react_agent_node(state)
            state.update(result1)
            tool_executor_node(state)
            state["iteration_count"] = result1["iteration_count"]
            
            result2 = react_agent_node(state)
            state.update(result2)
            tool_executor_node(state)
            assert state["analysis_result"] is not None
            
            state["iteration_count"] = result2["iteration_count"]
            result3 = react_agent_node(state)
            assert result3["done"] == True
    
    def test_parallel_tool_calls(self):
        state = create_initial_state(
            query="获取陕西省政策和山东负荷数据",
            provinces=["SN", "SD"],
            max_iterations=5,
        )
        
        mock_graph = MagicMock()
        mock_llm = MagicMock()
        
        mock_llm.invoke_with_tools.side_effect = [
            AIMessage(content="", tool_calls=[
                {"name": "retrieve_policy", "args": {"query": "电力政策"}, "id": "call_1"},
                {"name": "fetch_electricity_data", "args": {"province": "SN", "metric": "load"}, "id": "call_2"},
            ]),
            AIMessage(content="综合查询结果..."),
        ]
        
        mock_graph.llm_wrapper = mock_llm
        mock_graph.settings = MagicMock()
        mock_graph.settings.tools_enabled_list = None
        
        from app.agent.graph.handlers.iteration_control import IterationController
        mock_graph._iteration_controller = IterationController(max_iterations=5, timeout_seconds=30)
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            result1 = react_agent_node(state)
            assert len(result1["tool_calls"]) == 2
            
            state.update(result1)
            result2 = tool_executor_node(state)
            assert len(result2["tool_results"]) == 2
    
    def test_three_tool_chain(self):
        state = create_initial_state(
            query="查询政策、获取数据并分析",
            provinces=["SN"],
            max_iterations=6,
        )
        
        mock_graph = MagicMock()
        mock_llm = MagicMock()
        
        mock_llm.invoke_with_tools.side_effect = [
            AIMessage(content="", tool_calls=[
                {"name": "retrieve_policy", "args": {}, "id": "call_1"}
            ]),
            AIMessage(content="", tool_calls=[
                {"name": "fetch_electricity_data", "args": {}, "id": "call_2"}
            ]),
            AIMessage(content="", tool_calls=[
                {"name": "analyze_statistics", "args": {}, "id": "call_3"}
            ]),
            AIMessage(content="完整分析结果..."),
        ]
        
        mock_graph.llm_wrapper = mock_llm
        mock_graph.settings = MagicMock()
        mock_graph.settings.tools_enabled_list = None
        
        from app.agent.graph.handlers.iteration_control import IterationController
        mock_graph._iteration_controller = IterationController(max_iterations=6, timeout_seconds=30)
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            iterations = 0
            while iterations < 4:
                result = react_agent_node(state)
                state.update(result)
                if result.get("done"):
                    break
                tool_result = tool_executor_node(state)
                state.update(tool_result)
                state["iteration_count"] = result["iteration_count"]
                iterations += 1
            
            assert state["policy_chunks"] is not None and len(state["policy_chunks"]) > 0
            assert state["electricity_data"] is not None
            assert state["analysis_result"] is not None
            assert state.get("done") == True
    
    def test_tool_sequence_with_args_passing(self):
        state = create_initial_state(
            query="分析负荷趋势",
            provinces=["SN"],
            max_iterations=5,
        )
        
        mock_graph = MagicMock()
        mock_llm = MagicMock()
        
        mock_llm.invoke_with_tools.side_effect = [
            AIMessage(content="", tool_calls=[
                {"name": "fetch_electricity_data", "args": {"province": "SN", "metric": "load", "time_range": "24h"}, "id": "call_1"}
            ]),
            AIMessage(content="", tool_calls=[
                {"name": "analyze_statistics", "args": {"data": [], "analysis_type": "trend"}, "id": "call_2"}
            ]),
            AIMessage(content="分析完成"),
        ]
        
        mock_graph.llm_wrapper = mock_llm
        mock_graph.settings = MagicMock()
        mock_graph.settings.tools_enabled_list = None
        
        from app.agent.graph.handlers.iteration_control import IterationController
        mock_graph._iteration_controller = IterationController(max_iterations=5, timeout_seconds=30)
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            result1 = react_agent_node(state)
            state.update(result1)
            tool_executor_node(state)
            state["iteration_count"] = result1["iteration_count"]
            
            result2 = react_agent_node(state)
            state.update(result2)
            tool_executor_node(state)
            
            assert state["analysis_result"] is not None
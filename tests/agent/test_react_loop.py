import pytest
from unittest.mock import MagicMock, patch

from app.agent.graph.nodes.tool_executor_node import tool_executor_node
from app.agent.graph.state import ElectricityAgentState, create_initial_state


class TestToolExecutorNode:
    def test_tool_executor_no_calls(self):
        state = create_initial_state(
            query="电力交易规则",
            provinces=["SN"],
            max_iterations=5,
        )
        
        result = tool_executor_node(state)
        
        assert result["tool_results"] == []
    
    def test_tool_executor_retrieve_policy(self):
        state = create_initial_state(
            query="电力交易规则",
            provinces=["SN"],
            max_iterations=5,
        )
        state["tool_calls"] = [
            {"name": "retrieve_policy", "args": {"query": "电力交易规则", "provinces": ["SN"]}, "id": "call_1"}
        ]
        
        result = tool_executor_node(state)
        
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["tool_name"] == "retrieve_policy"
        assert result["tool_results"][0]["success"] == True
    
    def test_tool_executor_fetch_data(self):
        state = create_initial_state(
            query="负荷数据",
            provinces=["SN"],
            max_iterations=5,
        )
        state["tool_calls"] = [
            {"name": "fetch_electricity_data", "args": {"province": "SN", "metric": "load", "time_range": "24h"}, "id": "call_1"}
        ]
        
        result = tool_executor_node(state)
        
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["tool_name"] == "fetch_electricity_data"
        assert result["tool_results"][0]["success"] == True
    
    def test_tool_executor_analyze_statistics(self):
        state = create_initial_state(
            query="数据分析",
            provinces=["SN"],
            max_iterations=5,
        )
        state["electricity_data"] = {"data": [100, 150, 200, 250, 300]}
        state["tool_calls"] = [
            {"name": "analyze_statistics", "args": {"data": [], "analysis_type": "summary"}, "id": "call_1"}
        ]
        
        result = tool_executor_node(state)
        
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["success"] == True
    
    def test_tool_executor_unknown_tool(self):
        state = create_initial_state(
            query="测试",
            provinces=["SN"],
            max_iterations=5,
        )
        state["tool_calls"] = [
            {"name": "unknown_tool", "args": {}, "id": "call_1"}
        ]
        
        result = tool_executor_node(state)
        
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["success"] == False
    
    def test_tool_executor_clears_tool_calls(self):
        state = create_initial_state(
            query="电力交易规则",
            provinces=["SN"],
            max_iterations=5,
        )
        state["tool_calls"] = [
            {"name": "retrieve_policy", "args": {"query": "test", "provinces": ["SN"]}, "id": "call_1"}
        ]
        
        result = tool_executor_node(state)
        
        assert result["tool_calls"] == []
    
    def test_tool_executor_multiple_calls(self):
        state = create_initial_state(
            query="综合分析",
            provinces=["SN"],
            max_iterations=5,
        )
        state["tool_calls"] = [
            {"name": "retrieve_policy", "args": {"query": "test", "provinces": ["SN"]}, "id": "call_1"},
            {"name": "fetch_electricity_data", "args": {"province": "SN", "metric": "load"}, "id": "call_2"},
        ]
        
        result = tool_executor_node(state)
        
        assert len(result["tool_results"]) == 2
    
    def test_tool_executor_updates_state_data(self):
        state = create_initial_state(
            query="数据分析",
            provinces=["SN"],
            max_iterations=5,
        )
        state["tool_calls"] = [
            {"name": "fetch_electricity_data", "args": {"province": "SN", "metric": "load"}, "id": "call_1"}
        ]
        
        result = tool_executor_node(state)
        
        assert state["electricity_data"] is not None


class TestStateUpdate:
    def test_policy_chunks_updated(self):
        state = create_initial_state(
            query="政策查询",
            provinces=["SN"],
            max_iterations=5,
        )
        state["tool_calls"] = [
            {"name": "retrieve_policy", "args": {"query": "test", "provinces": ["SN"]}, "id": "call_1"}
        ]
        
        tool_executor_node(state)
        
        assert len(state["policy_chunks"]) > 0
    
    def test_analysis_result_updated(self):
        state = create_initial_state(
            query="分析",
            provinces=["SN"],
            max_iterations=5,
        )
        state["tool_calls"] = [
            {"name": "analyze_statistics", "args": {"data": [100, 200, 300], "analysis_type": "summary"}, "id": "call_1"}
        ]
        
        tool_executor_node(state)
        
        assert state["analysis_result"] is not None
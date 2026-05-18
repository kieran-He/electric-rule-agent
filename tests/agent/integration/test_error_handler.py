import pytest
import json
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from app.agent.graph.state import create_initial_state
from app.agent.graph.handlers.error_handler import ErrorHandler, handle_tool_error
from app.agent.graph.nodes.tool_executor_node import tool_executor_node
from app.agent.graph.nodes.react_agent_node import react_agent_node


class TestErrorHandler:
    def test_error_handler_retry_success(self):
        handler = ErrorHandler(max_retries=2, retry_delay=0.1)
        
        call_count = [0]
        
        def invoke_side_effect(args):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("第一次失败")
            return json.dumps({"success": True})
        
        mock_tool = MagicMock()
        mock_tool.invoke = invoke_side_effect
        
        success, result = handler.execute_with_retry(mock_tool, "test_tool", {})
        
        assert success == True
        assert call_count[0] == 2
    
    def test_error_handler_retry_exhausted(self):
        handler = ErrorHandler(max_retries=2, retry_delay=0.1)
        
        mock_tool = MagicMock()
        mock_tool.invoke = lambda args: (_ for _ in ()).throw(Exception("持续失败"))
        
        success, result = handler.execute_with_retry(mock_tool, "test_tool", {})
        
        assert success == False
        assert "error" in result
    
    def test_error_handler_no_retry_needed(self):
        handler = ErrorHandler(max_retries=2, retry_delay=0.1)
        
        mock_tool = MagicMock()
        mock_tool.invoke = lambda args: json.dumps({"success": True})
        
        success, result = handler.execute_with_retry(mock_tool, "test_tool", {})
        
        assert success == True
    
    def test_error_handler_exponential_backoff(self):
        handler = ErrorHandler(max_retries=2, retry_delay=1.0, exponential_backoff=True)
        
        assert handler.retry_delay == 1.0
        assert handler.exponential_backoff == True
    
    def test_degraded_response_creation(self):
        handler = ErrorHandler()
        
        response = handler.create_degraded_response(
            "test_tool",
            Exception("测试错误"),
            partial_data={"some_data": "value"}
        )
        
        assert response["error"] == True
        assert response["degraded"] == True
        assert response["tool_name"] == "test_tool"
        assert response["partial_data"] == {"some_data": "value"}
    
    def test_handle_tool_error_timeout(self):
        error = Exception("Timeout occurred")
        result = handle_tool_error(error, "test_tool", {"arg": "value"})
        
        assert result["error"] == "timeout"
        assert result["tool"] == "test_tool"
    
    def test_handle_tool_error_network(self):
        error = Exception("Network connection failed")
        result = handle_tool_error(error, "test_tool", {"arg": "value"})
        
        assert result["error"] == "network"
    
    def test_handle_tool_error_generic(self):
        error = Exception("Unknown error")
        result = handle_tool_error(error, "test_tool", {"arg": "value"})
        
        assert result["error"] == "Exception"
        assert "args" in result
    
    def test_tool_executor_unknown_tool_handling(self):
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
        assert len(result["errors"]) > 0
    
    def test_tool_executor_exception_handling(self):
        state = create_initial_state(
            query="测试",
            provinces=["SN"],
            max_iterations=5,
        )
        
        def failing_invoke(args):
            raise Exception("工具执行失败")
        
        with patch.dict("app.agent.graph.tools.tool_registry.ALL_TOOLS", {"retrieve_policy": MagicMock(invoke=failing_invoke)}):
            state["tool_calls"] = [
                {"name": "retrieve_policy", "args": {"query": "test"}, "id": "call_1"}
            ]
            
            result = tool_executor_node(state)
            
            assert len(result["tool_results"]) == 1
            assert result["tool_results"][0]["success"] == False
    
    def test_react_agent_error_handling(self):
        state = create_initial_state(
            query="测试",
            provinces=["SN"],
            max_iterations=5,
        )
        
        mock_graph = MagicMock()
        mock_llm = MagicMock()
        mock_llm.invoke_with_tools = lambda messages, tools: (_ for _ in ()).throw(Exception("LLM调用失败"))
        
        mock_graph.llm_wrapper = mock_llm
        mock_graph.settings = MagicMock()
        mock_graph.settings.tools_enabled_list = None
        
        from app.agent.graph.handlers.iteration_control import IterationController
        mock_graph._iteration_controller = IterationController(max_iterations=5, timeout_seconds=30)
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            result = react_agent_node(state)
            
            assert result["done"] == True
            assert len(result.get("errors", [])) > 0
    
    def test_tool_executor_partial_failure(self):
        state = create_initial_state(
            query="测试",
            provinces=["SN"],
            max_iterations=5,
        )
        state["tool_calls"] = [
            {"name": "retrieve_policy", "args": {"query": "test"}, "id": "call_1"},
            {"name": "unknown_tool", "args": {}, "id": "call_2"},
            {"name": "fetch_electricity_data", "args": {"province": "SN"}, "id": "call_3"},
        ]
        
        result = tool_executor_node(state)
        
        assert len(result["tool_results"]) == 3
        successes = [r["success"] for r in result["tool_results"]]
        assert True in successes
        assert False in successes
    
    def test_state_errors_accumulation(self):
        state = create_initial_state(
            query="测试",
            provinces=["SN"],
            max_iterations=5,
        )
        state["errors"] = [{"error": "previous_error"}]
        
        state["tool_calls"] = [
            {"name": "unknown_tool", "args": {}, "id": "call_1"}
        ]
        
        result = tool_executor_node(state)
        
        assert len(result["errors"]) >= 2
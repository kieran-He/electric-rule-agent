import pytest
import time
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from app.agent.graph.state import create_initial_state
from app.agent.graph.handlers.iteration_control import IterationController
from app.agent.graph.electricity_agent_graph import _should_continue


class TestIterationLimit:
    def test_iteration_controller_basic(self):
        controller = IterationController(max_iterations=3, timeout_seconds=30)
        
        state = {"iteration_count": 1, "done": False, "tool_results": []}
        
        should_continue, reason = controller.should_continue(state)
        assert should_continue == True
        assert reason == "continue"
    
    def test_iteration_controller_max_reached(self):
        controller = IterationController(max_iterations=3, timeout_seconds=30)
        
        state = {"iteration_count": 3, "done": False, "tool_results": []}
        
        should_continue, reason = controller.should_continue(state)
        assert should_continue == False
        assert reason == "max_iterations"
    
    def test_iteration_controller_timeout(self):
        controller = IterationController(max_iterations=5, timeout_seconds=30)
        
        state = {"iteration_count": 2, "done": False, "tool_results": []}
        
        should_continue, reason = controller.should_continue(state, elapsed_time=35)
        assert should_continue == False
        assert reason == "timeout"
    
    def test_iteration_controller_done_flag(self):
        controller = IterationController(max_iterations=5, timeout_seconds=30)
        
        state = {"iteration_count": 1, "done": True, "tool_results": []}
        
        should_continue, reason = controller.should_continue(state)
        assert should_continue == False
        assert reason == "done"
    
    def test_loop_detection_same_tool_repeated(self):
        controller = IterationController(max_iterations=10, loop_detection_window=3, timeout_seconds=30)
        
        state = {
            "iteration_count": 5,
            "done": False,
            "tool_results": [
                {"tool_name": "retrieve_policy"},
                {"tool_name": "retrieve_policy"},
                {"tool_name": "retrieve_policy"},
            ]
        }
        
        should_continue, reason = controller.should_continue(state)
        assert should_continue == False
        assert reason == "loop_detected"
    
    def test_loop_detection_different_tools(self):
        controller = IterationController(max_iterations=10, loop_detection_window=3, timeout_seconds=30)
        
        state = {
            "iteration_count": 5,
            "done": False,
            "tool_results": [
                {"tool_name": "retrieve_policy"},
                {"tool_name": "fetch_electricity_data"},
                {"tool_name": "analyze_statistics"},
            ]
        }
        
        should_continue, reason = controller.should_continue(state)
        assert should_continue == True
    
    def test_loop_detection_window_size(self):
        controller = IterationController(max_iterations=10, loop_detection_window=4, timeout_seconds=30)
        
        state = {
            "iteration_count": 5,
            "done": False,
            "tool_results": [
                {"tool_name": "retrieve_policy"},
                {"tool_name": "retrieve_policy"},
                {"tool_name": "retrieve_policy"},
            ]
        }
        
        should_continue, reason = controller.should_continue(state)
        assert should_continue == True
    
    def test_iteration_controller_stats(self):
        controller = IterationController(max_iterations=5, timeout_seconds=30)
        
        state = {
            "iteration_count": 3,
            "done": False,
            "tool_results": [
                {"tool_name": "retrieve_policy"},
                {"tool_name": "fetch_electricity_data"},
                {"tool_name": "retrieve_policy"},
            ]
        }
        
        stats = controller.get_stats(state)
        assert stats["iterations"] == 3
        assert stats["tools_called"] == 3
        assert stats["tool_counts"]["retrieve_policy"] == 2
        assert stats["tool_counts"]["fetch_electricity_data"] == 1
    
    def test_routing_stops_at_max_iterations(self):
        state = create_initial_state(
            query="测试",
            provinces=["SN"],
            max_iterations=3,
        )
        state["iteration_count"] = 3
        state["tool_calls"] = [{"name": "retrieve_policy", "args": {}, "id": "call_1"}]
        
        route = _should_continue(state)
        assert route == "end"
    
    def test_routing_stops_at_done(self):
        state = create_initial_state(
            query="测试",
            provinces=["SN"],
            max_iterations=5,
        )
        state["done"] = True
        state["iteration_count"] = 2
        state["tool_calls"] = [{"name": "retrieve_policy", "args": {}, "id": "call_1"}]
        
        route = _should_continue(state)
        assert route == "end"
    
    def test_routing_with_loop_detected(self):
        state = create_initial_state(
            query="测试",
            provinces=["SN"],
            max_iterations=10,
        )
        state["iteration_count"] = 5
        state["tool_results"] = [
            {"tool_name": "retrieve_policy"},
            {"tool_name": "retrieve_policy"},
            {"tool_name": "retrieve_policy"},
        ]
        state["tool_calls"] = [{"name": "retrieve_policy", "args": {}, "id": "call_1"}]
        
        mock_graph = MagicMock()
        controller = IterationController(max_iterations=10, loop_detection_window=3, timeout_seconds=30)
        mock_graph._iteration_controller = controller
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            route = _should_continue(state)
            assert route == "end"
    
    def test_check_iteration_limit_function(self):
        from app.agent.graph.handlers.iteration_control import check_iteration_limit
        
        assert check_iteration_limit(2, 5) == True
        assert check_iteration_limit(5, 5) == False
        assert check_iteration_limit(6, 5) == False
    
    def test_detect_repeated_tool_calls(self):
        from app.agent.graph.handlers.iteration_control import detect_repeated_tool_calls
        
        tool_calls = [
            {"name": "retrieve_policy", "args": {}},
            {"name": "retrieve_policy", "args": {}},
            {"name": "retrieve_policy", "args": {}},
        ]
        
        result = detect_repeated_tool_calls(tool_calls, window_size=3)
        assert result == "retrieve_policy"
        
        tool_calls_mixed = [
            {"name": "retrieve_policy", "args": {}},
            {"name": "fetch_electricity_data", "args": {}},
            {"name": "retrieve_policy", "args": {}},
        ]
        
        result = detect_repeated_tool_calls(tool_calls_mixed, window_size=3)
        assert result is None
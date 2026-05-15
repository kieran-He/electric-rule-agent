import logging
from typing import Dict, Any, List, Set, Optional

logger = logging.getLogger(__name__)


class IterationController:
    """Controls ReAct loop iterations and prevents infinite loops."""
    
    def __init__(
        self,
        max_iterations: int = 5,
        loop_detection_window: int = 3,
        timeout_seconds: int = 30,
    ):
        self.max_iterations = max_iterations
        self.loop_detection_window = loop_detection_window
        self.timeout_seconds = timeout_seconds
    
    def should_continue(
        self,
        state: Dict[str, Any],
        elapsed_time: float = 0,
    ) -> tuple[bool, str]:
        """
        Determine if the ReAct loop should continue.
        
        Args:
            state: Current agent state
            elapsed_time: Time elapsed since loop started
            
        Returns:
            Tuple of (should_continue, reason)
        """
        if state.get("done", False):
            logger.info("[IterationControl] Loop done flag set")
            return False, "done"
        
        iteration_count = state.get("iteration_count", 0)
        
        if iteration_count >= self.max_iterations:
            logger.info(f"[IterationControl] Max iterations reached: {iteration_count}")
            return False, "max_iterations"
        
        if elapsed_time > self.timeout_seconds:
            logger.warning(f"[IterationControl] Timeout exceeded: {elapsed_time}s")
            return False, "timeout"
        
        if self._detect_loop(state):
            logger.warning("[IterationControl] Loop detected")
            return False, "loop_detected"
        
        return True, "continue"
    
    def _detect_loop(self, state: Dict[str, Any]) -> bool:
        """
        Detect if the agent is stuck in a loop calling the same tool repeatedly.
        
        Args:
            state: Current agent state
            
        Returns:
            True if loop detected
        """
        tool_results = state.get("tool_results", [])
        if len(tool_results) < self.loop_detection_window:
            return False
        
        recent_tools = []
        for result in tool_results[-self.loop_detection_window:]:
            tool_name = result.get("tool_name", "")
            if tool_name:
                recent_tools.append(tool_name)
        
        if len(set(recent_tools)) == 1 and len(recent_tools) >= self.loop_detection_window:
            logger.warning(f"[IterationControl] Same tool repeated: {recent_tools[0]}")
            return True
        
        return False
    
    def get_stats(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get iteration statistics.
        
        Args:
            state: Current agent state
            
        Returns:
            Stats dict with iteration count, tool usage, etc.
        """
        tool_results = state.get("tool_results", [])
        tool_names = [r.get("tool_name", "") for r in tool_results]
        
        tool_counts: Dict[str, int] = {}
        for name in tool_names:
            tool_counts[name] = tool_counts.get(name, 0) + 1
        
        return {
            "iterations": state.get("iteration_count", 0),
            "max_iterations": self.max_iterations,
            "tools_called": len(tool_results),
            "tool_counts": tool_counts,
            "done": state.get("done", False),
        }


def check_iteration_limit(
    iteration: int,
    max_iterations: int,
) -> bool:
    """Check if iteration count exceeds limit."""
    return iteration < max_iterations


def detect_repeated_tool_calls(
    tool_calls: List[Dict[str, Any]],
    window_size: int = 3,
) -> Optional[str]:
    """
    Detect if the same tool is being called repeatedly.
    
    Args:
        tool_calls: List of recent tool calls
        window_size: Number of recent calls to check
        
    Returns:
        Name of tool if loop detected, None otherwise
    """
    if len(tool_calls) < window_size:
        return None
    
    recent_names = [tc.get("name", "") for tc in tool_calls[-window_size:]]
    if len(set(recent_names)) == 1:
        return recent_names[0]
    
    return None
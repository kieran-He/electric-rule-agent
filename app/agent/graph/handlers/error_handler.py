import logging
import time
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Handles tool execution errors with retry logic."""
    
    def __init__(
        self,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
    
    def execute_with_retry(
        self,
        func: Callable,
        tool_name: str,
        args: Dict[str, Any],
    ) -> tuple[bool, Any]:
        """
        Execute function with retry on failure.
        
        Args:
            func: Function to execute
            tool_name: Name of tool for logging
            args: Arguments to pass to function
            
        Returns:
            Tuple of (success, result_or_error)
        """
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                result = func.invoke(args)
                return True, result
                
            except Exception as e:
                last_error = e
                retries += 1
                
                if retries <= self.max_retries:
                    delay = self.retry_delay
                    if self.exponential_backoff:
                        delay = self.retry_delay * (2 ** (retries - 1))
                    
                    logger.warning(
                        f"[ErrorHandler] Tool {tool_name} failed, retry {retries}/{self.max_retries} "
                        f"after {delay}s delay. Error: {str(e)[:100]}"
                    )
                    time.sleep(delay)
        
        logger.error(f"[ErrorHandler] Tool {tool_name} failed after {retries} retries: {last_error}")
        return False, {"error": str(last_error), "tool": tool_name}
    
    def create_degraded_response(
        self,
        tool_name: str,
        error: Exception,
        partial_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Create a degraded response when tool fails completely.
        
        Args:
            tool_name: Name of failed tool
            error: The error that occurred
            partial_data: Any partial data that was collected
            
        Returns:
            Degraded response dict
        """
        return {
            "error": True,
            "tool_name": tool_name,
            "message": f"工具 {tool_name} 执行失败，请稍后重试",
            "error_detail": str(error)[:200],
            "partial_data": partial_data,
            "degraded": True,
        }


def handle_tool_error(
    error: Exception,
    tool_name: str,
    tool_args: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Handle a tool execution error.
    
    Args:
        error: The exception that occurred
        tool_name: Name of the tool
        tool_args: Arguments passed to the tool
        
    Returns:
        Error response dict
    """
    error_type = type(error).__name__
    
    if "timeout" in str(error).lower():
        return {
            "error": "timeout",
            "tool": tool_name,
            "message": f"工具 {tool_name} 执行超时",
        }
    
    if "connection" in str(error).lower() or "network" in str(error).lower():
        return {
            "error": "network",
            "tool": tool_name,
            "message": f"工具 {tool_name} 网络连接失败",
        }
    
    return {
        "error": error_type,
        "tool": tool_name,
        "message": f"工具 {tool_name} 执行失败: {str(error)[:100]}",
        "args": tool_args,
    }
import logging
from typing import List, Dict, Any, Optional

from langchain_core.tools import StructuredTool

from app.agent.graph.tools.policy_tool import retrieve_policy
from app.agent.graph.tools.data_tool import fetch_electricity_data
from app.agent.graph.tools.analysis_tool import analyze_statistics
from app.agent.graph.tools.web_tool import web_search

logger = logging.getLogger(__name__)

ALL_TOOLS = {
    "retrieve_policy": retrieve_policy,
    "fetch_electricity_data": fetch_electricity_data,
    "analyze_statistics": analyze_statistics,
    "web_search": web_search,
}

TOOL_METADATA = {
    "retrieve_policy": {
        "name": "retrieve_policy",
        "description": "检索电力市场相关政策文档和法规",
        "category": "knowledge",
    },
    "fetch_electricity_data": {
        "name": "fetch_electricity_data",
        "description": "获取电力市场数据（负荷、发电量、电价等）",
        "category": "data",
    },
    "analyze_statistics": {
        "name": "analyze_statistics",
        "description": "对电力数据进行统计分析",
        "category": "analysis",
    },
    "web_search": {
        "name": "web_search",
        "description": "网络搜索工具，用于获取最新新闻、政策动态、实时行情等时效性信息",
        "category": "knowledge",
    },
}


def get_all_tools() -> List[StructuredTool]:
    """
    Get all available tools as StructuredTool list.
    
    Returns:
        List of StructuredTool objects for LLM binding
    """
    return list(ALL_TOOLS.values())


def get_tools_by_names(tool_names: List[str]) -> List[StructuredTool]:
    """
    Get specific tools by their names.
    
    Args:
        tool_names: List of tool names to retrieve
        
    Returns:
        List of StructuredTool objects matching the names
    """
    tools = []
    for name in tool_names:
        if name in ALL_TOOLS:
            tools.append(ALL_TOOLS[name])
        else:
            logger.warning(f"[ToolRegistry] Unknown tool name: {name}")
    return tools


def get_tools_for_agent(enabled_tools: Optional[List[str]] = None) -> List[StructuredTool]:
    """
    Get tools configured for the agent.
    
    Args:
        enabled_tools: List of tool names to enable. If None, returns all tools.
        
    Returns:
        List of StructuredTool objects for agent use
    """
    if enabled_tools is None:
        return get_all_tools()
    
    return get_tools_by_names(enabled_tools)


def get_tool_info() -> Dict[str, Any]:
    """
    Get metadata about all available tools.
    
    Returns:
        Dict with tool names, descriptions, and categories
    """
    return {
        "tools": TOOL_METADATA,
        "count": len(ALL_TOOLS),
        "categories": {
            "knowledge": ["retrieve_policy", "web_search"],
            "data": ["fetch_electricity_data"],
            "analysis": ["analyze_statistics"],
        },
    }


def tool_exists(tool_name: str) -> bool:
    """Check if a tool exists in the registry."""
    return tool_name in ALL_TOOLS
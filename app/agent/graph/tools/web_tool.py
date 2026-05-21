import json
import logging
from typing import Dict, Any, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def web_search(query: str, provinces: List[str] = None) -> str:
    """
    网络搜索工具。用于补充知识库缺失的信息，特别是最新的新闻、政策动态、实时行情等。
    
    适用场景：
    - 用户询问"最新"、"最近"、"近期"等时效性信息
    - 知识库中没有相关省份的政策信息
    - 需要获取当前市场行情、实时数据
    
    Args:
        query: 搜索关键词
        provinces: 关注的省份代码列表，如["SN", "SD"]，用于针对性搜索
    
    Returns:
        搜索结果JSON字符串
    """
    try:
        from app.agent.graph.electricity_agent_graph import _get_current_instance
        
        graph_instance = _get_current_instance()
        
        if not graph_instance:
            logger.warning("[WebTool] No graph instance available")
            return json.dumps({
                "success": False,
                "output": "网络搜索服务暂不可用",
                "source": "error",
            })
        
        web_search_client = getattr(graph_instance, 'web_search_client', None)
        
        if not web_search_client or not web_search_client.is_available():
            logger.warning("[WebTool] Web search client not available")
            return json.dumps({
                "success": False,
                "output": "网络搜索服务暂不可用，请尝试其他查询方式",
                "source": "unavailable",
            })
        
        results = web_search_client.search(query)
        formatted = web_search_client.format_results_for_context(results)
        
        logger.info(f"[WebTool] Search completed, found {len(results)} results")
        
        return json.dumps({
            "success": True,
            "output": formatted,
            "confidence": 0.7,
            "result_count": len(results),
            "source": "web_search",
        })
        
    except Exception as e:
        logger.exception(f"[WebTool] Execution failed: {e}")
        return json.dumps({
            "success": False,
            "output": f"网络搜索失败: {str(e)[:100]}",
            "error": str(e),
            "source": "error",
        })
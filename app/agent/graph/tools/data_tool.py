import json
import logging
from typing import Dict, Any

from langchain_core.tools import tool

from app.agent.graph.tools.mock_data import generate_mock_electricity_data

logger = logging.getLogger(__name__)


@tool
def fetch_electricity_data(province: str, metric: str, time_range: str = "24h") -> str:
    """
    Fetch electricity market data for a specific province and metric.
    
    Use this tool when the user asks about numerical data like electricity
    load, generation output, prices, or renewable energy statistics.
    
    Args:
        province: Province code (e.g., "SN" for Shaanxi, "SX" for Shanxi)
        metric: Data type to fetch:
            - "load": Electricity demand/load data
            - "generation": Total generation output
            - "price": Market clearing prices
            - "new_energy": Renewable energy output (wind/solar)
        time_range: Time period for data:
            - "24h": Last 24 hours (default)
            - "7d": Last 7 days
            - "30d": Last 30 days
        
    Returns:
        JSON string containing data array, timestamps, and metadata
    """
    logger.info(f"[DataTool] Fetching data: province={province}, metric={metric}, time_range={time_range}")
    
    try:
        from app.agent.graph.electricity_agent_graph import _get_current_instance
        graph_instance = _get_current_instance()
        
        if graph_instance and graph_instance.data_adapter:
            data_adapter = graph_instance.data_adapter
            
            result = data_adapter.fetch_sync(
                province=province,
                metric=metric,
                time_range=time_range,
            )
            
            # 处理返回结果（可能是 Dict 或 list）
            if isinstance(result, dict):
                data = result.get("data", [])
                logger.info(f"[DataTool] Fetched {len(data)} data points from adapter (dict result)")
                if data:
                    return json.dumps(result, ensure_ascii=False)
            elif isinstance(result, list):
                logger.info(f"[DataTool] Fetched {len(result)} data points from adapter (list result)")
                if result:
                    return json.dumps({
                        "province": province,
                        "metric": metric,
                        "time_range": time_range,
                        "data": result,
                        "data_points": len(result),
                        "metadata": {"source": "adapter"},
                    }, ensure_ascii=False)
        
    except Exception as e:
        logger.warning(f"[DataTool] Adapter fetch failed: {e}, using mock data")
    
    mock_data = generate_mock_electricity_data(
        province=province,
        metric=metric,
        time_range=time_range,
        num_points=24,
    )
    logger.info(f"[DataTool] Using mock data: {mock_data['data_points']} points")
    
    return json.dumps(mock_data, ensure_ascii=False)
from __future__ import annotations

import logging
from typing import Dict, List, TYPE_CHECKING

from app.agent.graph.state import ElectricityAgentState

if TYPE_CHECKING:
    from app.agent.adapters.electricity_data_adapter import ElectricityDataAdapter

logger = logging.getLogger(__name__)


def parse_data_request(query: str, provinces: List[str]) -> Dict:
    province = provinces[0] if provinces else "SN"
    
    metric = "load"
    if "发电" in query or "发电量" in query:
        metric = "generation"
    elif "电价" in query or "价格" in query:
        metric = "price"
    elif "新能源" in query or "风电" in query or "光伏" in query:
        metric = "new_energy"
    elif "负荷" in query:
        metric = "load"
    
    time_range = "24h"
    if "昨日" in query or "昨天" in query:
        time_range = "24h"
    elif "本周" in query or "近7天" in query:
        time_range = "7d"
    elif "本月" in query or "近30天" in query:
        time_range = "30d"
    
    return {
        "province": province,
        "metric": metric,
        "time_range": time_range,
    }


async def data_fetcher_node_async(state: ElectricityAgentState) -> Dict:
    query = state["query"]
    provinces = state["provinces"]
    metadata = state.get("metadata", {})
    
    data_adapter = metadata.get("data_adapter")
    if not data_adapter:
        logger.warning("[DataFetcher] No data adapter available")
        return {
            "electricity_data": None,
            "tool_calls": state.get("tool_calls", []) + ["electricity_data"],
        }
    
    params = parse_data_request(query, provinces)
    
    try:
        data = await data_adapter.fetch(
            province=params["province"],
            metric=params["metric"],
            time_range=params["time_range"],
        )
        
        logger.info(f"[DataFetcher] Fetched {len(data)} data points")
        
        return {
            "electricity_data": {
                "province": params["province"],
                "metric": params["metric"],
                "time_range": params["time_range"],
                "data": data,
                "data_points": len(data),
            },
            "tool_calls": state.get("tool_calls", []) + ["electricity_data"],
            "metadata": {
                **metadata,
                "data_params": params,
            }
        }
    except Exception as e:
        logger.exception(f"[DataFetcher] Failed: {e}")
        return {
            "electricity_data": None,
            "tool_calls": state.get("tool_calls", []) + ["electricity_data"],
        }


def data_fetcher_node(state: ElectricityAgentState) -> Dict:
    query = state["query"]
    provinces = state["provinces"]
    
    from app.agent.graph.electricity_agent_graph import _get_current_instance
    graph_instance = _get_current_instance()
    
    if not graph_instance:
        logger.warning("[DataFetcher] No graph instance available")
        return {
            "electricity_data": None,
            "tool_calls": state.get("tool_calls", []) + ["electricity_data"],
        }
    
    data_adapter = graph_instance.data_adapter
    params = parse_data_request(query, provinces)
    
    try:
        data = data_adapter.fetch_sync(
            province=params["province"],
            metric=params["metric"],
            time_range=params["time_range"],
        )
        
        logger.info(f"[DataFetcher] Fetched {len(data)} data points")
        
        return {
            "electricity_data": {
                "province": params["province"],
                "metric": params["metric"],
                "time_range": params["time_range"],
                "data": data,
                "data_points": len(data),
            },
            "tool_calls": state.get("tool_calls", []) + ["electricity_data"],
        }
    except Exception as e:
        logger.exception(f"[DataFetcher] Failed: {e}")
        return {
            "electricity_data": None,
            "tool_calls": state.get("tool_calls", []) + ["electricity_data"],
        }
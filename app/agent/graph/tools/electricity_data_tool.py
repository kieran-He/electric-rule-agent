from typing import Dict, Any

from app.agent.graph.state import ElectricityAgentState


async def electricity_data_tool_node(
    state: ElectricityAgentState,
    province: str,
    metric: str,
    time_range: str = "24h",
) -> Dict[str, Any]:
    return {
        "province": province,
        "metric": metric,
        "time_range": time_range,
        "data": [],
        "data_points": 0,
    }
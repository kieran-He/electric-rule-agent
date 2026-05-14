import numpy as np
from typing import Dict, Any


async def analysis_tool_node(
    state: Dict[str, Any],
    data: Dict[str, Any],
    analysis_type: str = "summary",
) -> Dict[str, Any]:
    values = data.get("data", [])
    if not values:
        return {"error": "no_data"}
    
    arr = np.array(values)
    result = {}
    
    if analysis_type == "mean":
        result["mean"] = np.mean(arr)
        result["median"] = np.median(arr)
    elif analysis_type == "variance":
        result["variance"] = np.var(arr)
        result["std"] = np.std(arr)
        result["range"] = np.max(arr) - np.min(arr)
    elif analysis_type == "distribution":
        result["mean"] = np.mean(arr)
        result["std"] = np.std(arr)
        result["percentiles"] = {
            "p25": np.percentile(arr, 25),
            "p50": np.percentile(arr, 50),
            "p75": np.percentile(arr, 75),
        }
    elif analysis_type == "trend":
        result["growth_rate"] = (arr[-1] - arr[0]) / arr[0] * 100 if arr[0] != 0 else 0
        result["trend"] = "up" if result["growth_rate"] > 0 else "down"
    else:
        result["mean"] = np.mean(arr)
        result["std"] = np.std(arr)
        result["min"] = np.min(arr)
        result["max"] = np.max(arr)
    
    return result
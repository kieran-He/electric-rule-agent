import json
import logging
import numpy as np
from typing import List, Dict, Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def analyze_statistics(data: List[float], analysis_type: str = "summary") -> str:
    """
    Perform statistical analysis on electricity data.
    
    Use this tool to compute statistics, trends, or distributions
    from numerical electricity market data.
    
    Args:
        data: List of numerical values to analyze (e.g., prices, load values)
        analysis_type: Type of analysis to perform:
            - "summary": Basic stats (mean, std, min, max) [default]
            - "mean": Mean and median
            - "variance": Variance, std, and range
            - "distribution": Distribution with percentiles (p25, p50, p75)
            - "trend": Growth rate and trend direction
        
    Returns:
        JSON string containing analysis results
    """
    logger.info(f"[AnalysisTool] Analyzing {len(data)} data points, type={analysis_type}")
    
    if not data:
        return json.dumps({"error": "no_data", "message": "No data provided for analysis"})
    
    try:
        arr = np.array(data)
        result = {}
        
        if analysis_type == "mean":
            result["mean"] = float(np.mean(arr))
            result["median"] = float(np.median(arr))
            result["count"] = len(data)
            
        elif analysis_type == "variance":
            result["variance"] = float(np.var(arr))
            result["std"] = float(np.std(arr))
            result["range"] = float(np.max(arr) - np.min(arr))
            result["min"] = float(np.min(arr))
            result["max"] = float(np.max(arr))
            
        elif analysis_type == "distribution":
            result["mean"] = float(np.mean(arr))
            result["std"] = float(np.std(arr))
            result["percentiles"] = {
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
            }
            result["histogram_bins"] = 10
            
        elif analysis_type == "trend":
            if len(data) >= 2 and arr[0] != 0:
                growth_rate = float((arr[-1] - arr[0]) / arr[0] * 100)
                result["growth_rate"] = round(growth_rate, 2)
                result["trend"] = "up" if growth_rate > 0 else "down"
                result["start_value"] = float(arr[0])
                result["end_value"] = float(arr[-1])
            else:
                result["growth_rate"] = 0.0
                result["trend"] = "stable"
                
        else:
            result["mean"] = float(np.mean(arr))
            result["std"] = float(np.std(arr))
            result["min"] = float(np.min(arr))
            result["max"] = float(np.max(arr))
            result["median"] = float(np.median(arr))
            result["count"] = len(data)
        
        result["analysis_type"] = analysis_type
        logger.info(f"[AnalysisTool] Analysis complete: {result}")
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.exception(f"[AnalysisTool] Analysis failed: {e}")
        return json.dumps({"error": str(e), "message": "Analysis failed"})


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
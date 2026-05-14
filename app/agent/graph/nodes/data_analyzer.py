from __future__ import annotations

import logging
import numpy as np
from typing import Dict, List

from app.agent.graph.state import ElectricityAgentState

logger = logging.getLogger(__name__)


def detect_analysis_type(query: str) -> str:
    if "均值" in query or "平均" in query:
        return "mean"
    elif "方差" in query or "标准差" in query:
        return "variance"
    elif "分布" in query or "百分比" in query:
        return "distribution"
    elif "趋势" in query or "增长" in query:
        return "trend"
    else:
        return "summary"


def analyze_data(data: List[float], analysis_type: str) -> Dict:
    if not data:
        return {"error": "no_data"}
    
    arr = np.array(data)
    result = {}
    
    if analysis_type == "mean":
        result["mean"] = float(np.mean(arr))
        result["median"] = float(np.median(arr))
    elif analysis_type == "variance":
        result["variance"] = float(np.var(arr))
        result["std"] = float(np.std(arr))
        result["range"] = float(np.max(arr) - np.min(arr))
    elif analysis_type == "distribution":
        result["mean"] = float(np.mean(arr))
        result["std"] = float(np.std(arr))
        result["percentiles"] = {
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
        }
    elif analysis_type == "trend":
        if len(arr) > 1 and arr[0] != 0:
            result["growth_rate"] = float((arr[-1] - arr[0]) / arr[0] * 100)
        else:
            result["growth_rate"] = 0.0
        result["trend"] = "up" if result["growth_rate"] > 0 else "down"
    else:
        result["mean"] = float(np.mean(arr))
        result["std"] = float(np.std(arr))
        result["min"] = float(np.min(arr))
        result["max"] = float(np.max(arr))
        result["count"] = len(data)
    
    return result


def data_analyzer_node(state: ElectricityAgentState) -> Dict:
    electricity_data = state.get("electricity_data")
    query = state["query"]
    
    if not electricity_data:
        logger.warning("[DataAnalyzer] No electricity data available")
        return {
            "analysis_result": {"error": "no_data"},
            "tool_calls": state.get("tool_calls", []) + ["analysis"],
        }
    
    data = electricity_data.get("data", [])
    if not data:
        logger.warning("[DataAnalyzer] Empty data")
        return {
            "analysis_result": {"error": "empty_data"},
            "tool_calls": state.get("tool_calls", []) + ["analysis"],
        }
    
    analysis_type = detect_analysis_type(query)
    result = analyze_data(data, analysis_type)
    
    logger.info(f"[DataAnalyzer] Analysis type: {analysis_type}, result keys: {list(result.keys())}")
    
    return {
        "analysis_result": result,
        "tool_calls": state.get("tool_calls", []) + ["analysis"],
    }
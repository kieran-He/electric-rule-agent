import json
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock


class MockToolRegistry:
    MOCK_POLICY_RESULT = [
        {"content": "陕西省电力市场交易规则规定...", "source": "陕西省电力交易中心", "score": 0.85},
        {"content": "市场主体包括发电企业、售电公司...", "source": "陕西省能源局", "score": 0.80},
    ]
    
    MOCK_DATA_RESULT = {
        "province": "SN",
        "metric": "load",
        "data_points": 24,
        "data": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
        "metadata": {"source": "mock", "time_range": "24h"},
    }
    
    MOCK_ANALYSIS_RESULT = {
        "mean": 1250.0,
        "median": 1250.0,
        "std": 158.11,
        "min": 1000.0,
        "max": 1500.0,
        "count": 6,
    }
    
    @staticmethod
    def get_mock_tool(name: str) -> MagicMock:
        tool = MagicMock()
        tool.name = name
        
        if name == "retrieve_policy":
            tool.invoke = MockToolRegistry._mock_retrieve_policy_invoke
        elif name == "fetch_electricity_data":
            tool.invoke = MockToolRegistry._mock_fetch_data_invoke
        elif name == "analyze_statistics":
            tool.invoke = MockToolRegistry._mock_analyze_invoke
        else:
            tool.invoke = lambda args: json.dumps({"error": "unknown_tool"})
        
        return tool
    
    @staticmethod
    def _mock_retrieve_policy_invoke(args: Dict[str, Any]) -> str:
        return json.dumps(MockToolRegistry.MOCK_POLICY_RESULT)
    
    @staticmethod
    def _mock_fetch_data_invoke(args: Dict[str, Any]) -> str:
        result = MockToolRegistry.MOCK_DATA_RESULT.copy()
        result["province"] = args.get("province", "SN")
        result["metric"] = args.get("metric", "load")
        return json.dumps(result)
    
    @staticmethod
    def _mock_analyze_invoke(args: Dict[str, Any]) -> str:
        analysis_type = args.get("analysis_type", "summary")
        result = MockToolRegistry.MOCK_ANALYSIS_RESULT.copy()
        result["analysis_type"] = analysis_type
        return json.dumps(result)


def mock_retrieve_policy(args: Dict[str, Any]) -> str:
    return MockToolRegistry._mock_retrieve_policy_invoke(args)


def mock_fetch_data(args: Dict[str, Any]) -> str:
    return MockToolRegistry._mock_fetch_data_invoke(args)


def mock_analyze(args: Dict[str, Any]) -> str:
    return MockToolRegistry._mock_analyze_invoke(args)
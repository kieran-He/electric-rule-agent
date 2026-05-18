import json
import pytest
from unittest.mock import MagicMock, patch

from app.agent.graph.tools.policy_tool import retrieve_policy
from app.agent.graph.tools.data_tool import fetch_electricity_data
from app.agent.graph.tools.analysis_tool import analyze_statistics
from app.agent.graph.tools.tool_registry import (
    get_all_tools,
    get_tools_by_names,
    get_tools_for_agent,
    tool_exists,
    ALL_TOOLS,
)


class TestPolicyTool:
    def test_retrieve_policy_returns_json_string(self):
        result = retrieve_policy.invoke({"query": "电力交易规则", "provinces": ["SN"]})
        
        assert isinstance(result, str)
        
        parsed = json.loads(result)
        assert isinstance(parsed, list)
    
    def test_retrieve_policy_with_mock_data(self):
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=None):
            result = retrieve_policy.invoke({"query": "电力交易规则", "provinces": ["SN"]})
            
            parsed = json.loads(result)
            assert len(parsed) > 0
            assert "content" in parsed[0]
    
    def test_retrieve_policy_with_orchestrator(self):
        mock_graph = MagicMock()
        mock_orchestrator = MagicMock()
        
        mock_chunk = MagicMock()
        mock_chunk.text = "陕西省电力市场交易规则规定..."
        mock_chunk.metadata = {"source_name": "陕西省电力交易中心", "title_path": "交易规则"}
        mock_chunk.score = 0.85
        
        mock_orchestrator._retrieve.return_value = ([mock_chunk], ["SN"], {"quality": "high"})
        mock_graph.orchestrator = mock_orchestrator
        
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=mock_graph):
            result = retrieve_policy.invoke({"query": "电力交易规则", "provinces": ["SN"]})
            
            parsed = json.loads(result)
            assert len(parsed) > 0
            assert "陕西省电力市场交易规则" in parsed[0]["content"]


class TestDataTool:
    def test_fetch_electricity_data_returns_json_string(self):
        result = fetch_electricity_data.invoke({
            "province": "SN",
            "metric": "load",
            "time_range": "24h"
        })
        
        assert isinstance(result, str)
        
        parsed = json.loads(result)
        assert "data" in parsed
        assert "data_points" in parsed
        assert parsed["province"] == "SN"
    
    def test_fetch_electricity_data_with_mock_data(self):
        with patch("app.agent.graph.electricity_agent_graph._get_current_instance", return_value=None):
            result = fetch_electricity_data.invoke({
                "province": "SN",
                "metric": "load",
                "time_range": "24h"
            })
            
            parsed = json.loads(result)
            assert parsed["metadata"]["source"] == "mock"
            assert len(parsed["data"]) > 0
    
    def test_fetch_electricity_data_different_metrics(self):
        metrics = ["load", "generation", "price", "new_energy"]
        
        for metric in metrics:
            result = fetch_electricity_data.invoke({
                "province": "SN",
                "metric": metric,
                "time_range": "24h"
            })
            
            parsed = json.loads(result)
            assert parsed["metric"] == metric


class TestAnalysisTool:
    def test_analyze_statistics_summary(self):
        data = [100.0, 150.0, 200.0, 250.0, 300.0]
        
        result = analyze_statistics.invoke({
            "data": data,
            "analysis_type": "summary"
        })
        
        parsed = json.loads(result)
        assert "mean" in parsed
        assert "std" in parsed
        assert "min" in parsed
        assert "max" in parsed
        assert parsed["count"] == 5
    
    def test_analyze_statistics_mean(self):
        data = [100.0, 150.0, 200.0]
        
        result = analyze_statistics.invoke({
            "data": data,
            "analysis_type": "mean"
        })
        
        parsed = json.loads(result)
        assert "mean" in parsed
        assert "median" in parsed
        assert parsed["mean"] == 150.0
    
    def test_analyze_statistics_trend(self):
        data = [100.0, 120.0, 140.0, 160.0]
        
        result = analyze_statistics.invoke({
            "data": data,
            "analysis_type": "trend"
        })
        
        parsed = json.loads(result)
        assert "growth_rate" in parsed
        assert "trend" in parsed
        assert parsed["trend"] == "up"
    
    def test_analyze_statistics_no_data(self):
        result = analyze_statistics.invoke({
            "data": [],
            "analysis_type": "summary"
        })
        
        parsed = json.loads(result)
        assert "error" in parsed
    
    def test_analyze_statistics_distribution(self):
        data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        result = analyze_statistics.invoke({
            "data": data,
            "analysis_type": "distribution"
        })
        
        parsed = json.loads(result)
        assert "percentiles" in parsed
        assert "p25" in parsed["percentiles"]
        assert "p50" in parsed["percentiles"]
        assert "p75" in parsed["percentiles"]


class TestToolRegistry:
    def test_get_all_tools(self):
        tools = get_all_tools()
        
        assert len(tools) == 4
        assert all(hasattr(t, 'name') for t in tools)
    
    def test_get_tools_by_names(self):
        tools = get_tools_by_names(["retrieve_policy", "fetch_electricity_data"])
        
        assert len(tools) == 2
    
    def test_get_tools_by_names_unknown(self):
        tools = get_tools_by_names(["retrieve_policy", "unknown_tool"])
        
        assert len(tools) == 1
    
    def test_get_tools_for_agent(self):
        tools = get_tools_for_agent(["retrieve_policy"])
        
        assert len(tools) == 1
    
    def test_get_tools_for_agent_none(self):
        tools = get_tools_for_agent(None)
        
        assert len(tools) == 4
    
    def test_tool_exists(self):
        assert tool_exists("retrieve_policy")
        assert tool_exists("fetch_electricity_data")
        assert tool_exists("web_search")
        assert not tool_exists("unknown_tool")
    
    def test_all_tools_dict(self):
        assert "retrieve_policy" in ALL_TOOLS
        assert "fetch_electricity_data" in ALL_TOOLS
        assert "analyze_statistics" in ALL_TOOLS
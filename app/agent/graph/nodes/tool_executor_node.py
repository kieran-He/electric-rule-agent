import json
import logging
from typing import Dict, Any

from app.agent.graph.state import ElectricityAgentState
from app.agent.graph.tools.tool_registry import ALL_TOOLS

logger = logging.getLogger(__name__)


def tool_executor_node(state: ElectricityAgentState) -> Dict[str, Any]:
    """
    Execute tools based on tool_calls in state.
    
    Runs each tool, captures results, and updates state.
    """
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        logger.warning("[ToolExecutor] No tool calls to execute")
        return {"tool_results": [], "tool_calls": []}
    
    logger.info(f"[ToolExecutor] Executing {len(tool_calls)} tools")
    
    tool_results = []
    errors = []
    
    for tc in tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_id = tc.get("id", "")
        
        logger.info(f"[ToolExecutor] Running tool: {tool_name} with args: {tool_args}")
        
        if tool_name not in ALL_TOOLS:
            logger.warning(f"[ToolExecutor] Unknown tool: {tool_name}")
            errors.append({
                "tool": tool_name,
                "error": "unknown_tool",
                "message": f"Tool '{tool_name}' not found in registry",
            })
            tool_results.append({
                "tool_name": tool_name,
                "tool_call_id": tool_id,
                "output": json.dumps({"error": "unknown_tool"}),
                "success": False,
            })
            continue
        
        tool_func = ALL_TOOLS[tool_name]
        
        try:
            if tool_name == "retrieve_policy":
                query = tool_args.get("query", state.get("query", ""))
                provinces = tool_args.get("provinces", state.get("provinces", ["SN"]))
                output = tool_func.invoke({"query": query, "provinces": provinces})
                
            elif tool_name == "fetch_electricity_data":
                province = tool_args.get("province", state.get("provinces", ["SN"])[0])
                metric = tool_args.get("metric", "load")
                time_range = tool_args.get("time_range", "24h")
                output = tool_func.invoke({
                    "province": province,
                    "metric": metric,
                    "time_range": time_range,
                })
                
            elif tool_name == "analyze_statistics":
                data = tool_args.get("data", [])
                if not data and state.get("electricity_data"):
                    data = state["electricity_data"].get("data", [])
                analysis_type = tool_args.get("analysis_type", "summary")
                output = tool_func.invoke({
                    "data": data,
                    "analysis_type": analysis_type,
                })
                
            else:
                output = tool_func.invoke(tool_args)
            
            logger.info(f"[ToolExecutor] Tool {tool_name} success, output length: {len(output) if output else 0}")
            
            tool_results.append({
                "tool_name": tool_name,
                "tool_call_id": tool_id,
                "output": output,
                "success": True,
            })
            
            _update_state_data(state, tool_name, output)
            
        except Exception as e:
            logger.exception(f"[ToolExecutor] Tool {tool_name} failed: {e}")
            
            errors.append({
                "tool": tool_name,
                "error": str(e),
                "args": tool_args,
            })
            
            tool_results.append({
                "tool_name": tool_name,
                "tool_call_id": tool_id,
                "output": json.dumps({"error": str(e)}),
                "success": False,
            })
    
    return {
        "tool_results": tool_results,
        "tool_calls": [],
        "errors": state.get("errors", []) + errors,
    }


def _update_state_data(state: ElectricityAgentState, tool_name: str, output: str) -> None:
    try:
        result_data = json.loads(output) if output else {}
        
        if tool_name == "retrieve_policy":
            if isinstance(result_data, list):
                state["policy_chunks"] = result_data
            elif isinstance(result_data, dict) and "chunks" in result_data:
                state["policy_chunks"] = result_data["chunks"]
                
        elif tool_name == "fetch_electricity_data":
            state["electricity_data"] = result_data
            if isinstance(result_data, dict) and result_data.get("chart_path"):
                chart_paths = state.get("chart_paths", [])
                chart_path = result_data.get("chart_path")
                if chart_path and chart_path not in chart_paths:
                    chart_paths.append(chart_path)
                    state["chart_paths"] = chart_paths
                    logger.info(f"[ToolExecutor] Added chart_path: {chart_path}, total: {len(chart_paths)}")
            
        elif tool_name == "analyze_statistics":
            state["analysis_result"] = result_data
            
    except json.JSONDecodeError:
        logger.warning(f"[ToolExecutor] Could not parse output from {tool_name}")
    except Exception as e:
        logger.warning(f"[ToolExecutor] Could not update state from {tool_name}: {e}")
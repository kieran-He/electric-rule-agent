import json
import logging
from typing import Dict, Any

from app.agent.graph.state import ElectricityAgentState
from app.agent.graph.tools.tool_registry import ALL_TOOLS

logger = logging.getLogger(__name__)

# 信息充足判断阈值
SUFFICIENCY_THRESHOLDS = {
    "policy_output_min_length": 1500,  # 政策检索结果最小长度
    "web_output_min_length": 500,      # 网络搜索结果最小长度
    "max_iterations_before_force_stop": 2,  # 强制停止的最大迭代次数
}


def tool_executor_node(state: ElectricityAgentState) -> Dict[str, Any]:
    """
    Execute tools based on tool_calls in state.
    
    Runs each tool, captures results, and updates state.
    Also checks information sufficiency for early termination.
    """
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        logger.warning("[ToolExecutor] No tool calls to execute")
        return {"tool_results": [], "tool_calls": []}
    
    logger.info(f"[ToolExecutor] Executing {len(tool_calls)} tools")
    
    tool_results = []
    errors = []
    policy_chunks = []
    electricity_data = None
    chart_paths = []
    retrieval_quality = None
    need_web_search = False
    
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
                query = state.get("query", "")
                provinces = tool_args.get("provinces", state.get("provinces", ["SN"]))
                show_chunks = state.get("context", {}).get("show_chunks", True)
                output = tool_func.invoke({"query": query, "provinces": provinces, "show_chunks": show_chunks})
                
                try:
                    result_data = json.loads(output) if output else {}
                    if isinstance(result_data, dict) and "chunks" in result_data:
                        policy_chunks = result_data["chunks"]
                        retrieval_quality = result_data.get("quality")
                        formatted_chunks = result_data.get("formatted_chunks", "")
                        if retrieval_quality:
                            if retrieval_quality.get("is_low_quality") or retrieval_quality.get("chunk_count", 0) < 3:
                                need_web_search = True
                                logger.info(f"[ToolExecutor] Low quality retrieval, suggesting web_search: {retrieval_quality}")
                        if formatted_chunks:
                            output = formatted_chunks
                except json.JSONDecodeError:
                    pass
                
            elif tool_name == "fetch_electricity_data":
                province = tool_args.get("province", state.get("provinces", ["SN"])[0])
                metric = tool_args.get("metric", "load")
                time_range = tool_args.get("time_range", "24h")
                output = tool_func.invoke({
                    "province": province,
                    "metric": metric,
                    "time_range": time_range,
                })
                
                try:
                    result_data = json.loads(output) if output else {}
                    electricity_data = result_data
                    if isinstance(result_data, dict) and result_data.get("chart_path"):
                        chart_paths.append(result_data.get("chart_path"))
                except json.JSONDecodeError:
                    pass
                
            elif tool_name == "analyze_statistics":
                data = tool_args.get("data", [])
                if not data and state.get("electricity_data"):
                    data = state["electricity_data"].get("data", [])
                analysis_type = tool_args.get("analysis_type", "summary")
                output = tool_func.invoke({
                    "data": data,
                    "analysis_type": analysis_type,
                })
                
            elif tool_name == "web_search":
                query = state.get("query", "")
                provinces = tool_args.get("provinces", state.get("provinces", ["SN"]))
                output = tool_func.invoke({"query": query, "provinces": provinces})
                
            else:
                output = tool_func.invoke(tool_args)
            
            logger.info(f"[ToolExecutor] Tool {tool_name} success, output length: {len(output) if output else 0}")
            
            tool_results.append({
                "tool_name": tool_name,
                "tool_call_id": tool_id,
                "output": output,
                "success": True,
            })
            
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
    
    # 检查信息充足度
    all_tool_results = state.get("tool_results", []) + tool_results
    iteration_count = state.get("iteration_count", 0)
    
    sufficiency_info = _check_sufficiency(all_tool_results, iteration_count)
    
    result = {
        "tool_results": tool_results,
        "tool_calls": [],
        "errors": state.get("errors", []) + errors,
        "policy_chunks": policy_chunks,
        "electricity_data": electricity_data,
        "chart_paths": chart_paths,
        "retrieval_quality": retrieval_quality,
        "need_web_search": need_web_search,
    }
    
    if sufficiency_info.get("sufficient"):
        logger.info(f"[ToolExecutor] Information sufficient: {sufficiency_info.get('reason')}")
        result["sufficient_info"] = True
        result["sufficiency_reason"] = sufficiency_info.get("reason")
    
    return result


def _check_sufficiency(tool_results: list, iteration_count: int) -> Dict[str, Any]:
    """
    Check if the information collected is sufficient to answer the question.
    
    Returns:
        Dict with 'sufficient' bool and 'reason' string
    """
    # 收集各工具的结果长度
    policy_length = 0
    web_length = 0
    data_available = False
    tools_used = set()
    
    for result in tool_results:
        if result.get("success"):
            tool_name = result.get("tool_name", "")
            output = result.get("output", "")
            output_len = len(output) if output else 0
            tools_used.add(tool_name)
            
            if tool_name == "retrieve_policy":
                policy_length += output_len
            elif tool_name == "web_search":
                web_length += output_len
            elif tool_name == "fetch_electricity_data":
                data_available = output_len > 0
    
    # 判断条件
    reasons = []
    
    # 条件1: 政策检索结果足够长
    if policy_length >= SUFFICIENCY_THRESHOLDS["policy_output_min_length"]:
        reasons.append(f"政策检索结果充足({policy_length}字符)")
    
    # 条件2: 已执行了政策检索和网络搜索
    if "retrieve_policy" in tools_used and "web_search" in tools_used:
        reasons.append("已完成政策检索和网络搜索")
    
    # 条件3: 政策+网络结果总量足够
    total_length = policy_length + web_length
    if total_length >= 2000:
        reasons.append(f"总信息量充足({total_length}字符)")
    
    # 条件4: 迭代次数达到阈值，强制停止
    if iteration_count >= SUFFICIENCY_THRESHOLDS["max_iterations_before_force_stop"]:
        reasons.append(f"已达到{iteration_count}轮迭代，建议生成答案")
    
    # 判断是否充足
    sufficient = len(reasons) > 0
    
    return {
        "sufficient": sufficient,
        "reason": "; ".join(reasons) if reasons else "信息不足",
        "policy_length": policy_length,
        "web_length": web_length,
        "tools_used": list(tools_used),
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
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from app.agent.graph.state import ElectricityAgentState
from app.agent.graph.tools.tool_registry import get_tools_for_agent

logger = logging.getLogger(__name__)

REACT_SYSTEM_PROMPT = """你是一个电力市场分析助手，能够使用工具来回答用户问题。

可用工具:
1. retrieve_policy: 检索电力市场政策文档和法规
2. fetch_electricity_data: 获取电力数据（负荷、发电量、电价等）
3. analyze_statistics: 对数据进行统计分析

数据可用性:
- fetch_electricity_data 目前只支持陕西省(SN)的数据查询
- 如果用户询问其他省份的数据，请告知当前只支持陕西数据

工作流程:
1. 分析用户问题，确定需要使用哪些工具
2. 按顺序调用必要的工具获取信息
3. 综合工具结果，生成完整回答

注意事项:
- 如果问题涉及政策法规，使用 retrieve_policy
- 如果问题涉及陕西电力数据，使用 fetch_electricity_data（province参数设为"SN"）
- 如果需要对数据进行分析，使用 analyze_statistics
- 可以组合使用多个工具
- 当获取到足够信息后，直接给出答案，不要再调用工具"""


def _build_messages(state: ElectricityAgentState) -> List:
    messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)]
    
    for msg in state.get("messages", []):
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        elif isinstance(msg, HumanMessage):
            messages.append(msg)
        elif isinstance(msg, AIMessage):
            messages.append(msg)
        elif isinstance(msg, SystemMessage):
            continue
        else:
            messages.append(HumanMessage(content=str(msg)))
    
    # 构建工具调用和结果消息链
    if state.get("tool_results") and state.get("last_tool_calls"):
        # 先添加包含 tool_calls 的 AIMessage
        tool_calls_for_msg = []
        for tc in state["last_tool_calls"]:
            tool_calls_for_msg.append({
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
                "id": tc.get("id", f"tool_{tc.get('name', 'unknown')}"),
            })
        
        messages.append(AIMessage(content="", tool_calls=tool_calls_for_msg))
        
        # 然后添加 ToolMessage 结果
        for result in state["tool_results"]:
            tool_name = result.get("tool_name", "unknown")
            tool_output = result.get("output", "")
            tool_call_id = result.get("tool_call_id", f"tool_{tool_name}")
            messages.append(
                ToolMessage(
                    content=f"工具 {tool_name} 返回结果:\n{tool_output}",
                    tool_call_id=tool_call_id,
                )
            )
    
    if not state.get("messages") or state["iteration_count"] == 0:
        query = state["query"]
        provinces = state.get("provinces", ["SN"])
        province_str = ", ".join(provinces)
        
        user_prompt = f"用户问题: {query}\n关注省份: {province_str}"
        messages.append(HumanMessage(content=user_prompt))
    
    return messages


def react_agent_node(state: ElectricityAgentState) -> Dict[str, Any]:
    """
    ReAct agent node: decides next action based on current state.
    
    Either calls a tool or produces final answer.
    """
    logger.info(f"[ReActAgent] Iteration {state['iteration_count']}/{state['max_iterations']}")
    
    thoughts = state.get("thoughts", [])
    thoughts.append({
        "iteration": state["iteration_count"],
        "phase": "thinking",
        "timestamp": datetime.now().isoformat(),
    })
    
    try:
        from app.agent.graph.electricity_agent_graph import _get_current_instance
        graph_instance = _get_current_instance()
        
        if not graph_instance:
            logger.error("[ReActAgent] No graph instance available")
            return {
                "answer": "系统错误：无法访问分析引擎",
                "done": True,
                "errors": [{"error": "no_graph_instance"}],
            }
        
        llm_wrapper = graph_instance.llm_wrapper
        settings = graph_instance.settings
        
        enabled_tools = getattr(settings, 'tools_enabled_list', None)
        if enabled_tools is None:
            tools_str = getattr(settings, 'tools_enabled', None)
            if tools_str:
                enabled_tools = [t.strip() for t in tools_str.split(',') if t.strip()]
        tools = get_tools_for_agent(enabled_tools)
        
        messages = _build_messages(state)
        
        logger.info(f"[ReActAgent] Calling LLM with {len(messages)} messages, {len(tools)} tools")
        
        response = llm_wrapper.invoke_with_tools(messages, tools)
        
        logger.info(f"[ReActAgent] Response type: {type(response).__name__}")
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = []
            for tc in response.tool_calls:
                if isinstance(tc, dict):
                    tool_calls.append({
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                        "id": tc.get("id", ""),
                    })
                else:
                    tool_calls.append({
                        "name": getattr(tc, 'name', ''),
                        "args": getattr(tc, 'args', {}),
                        "id": getattr(tc, 'id', ''),
                    })
            
            logger.info(f"[ReActAgent] Tool calls: {[tc['name'] for tc in tool_calls]}")
            
            thoughts.append({
                "iteration": state["iteration_count"],
                "phase": "tool_call",
                "tools": [tc["name"] for tc in tool_calls],
            })
            
            return {
                "tool_calls": tool_calls,
                "last_tool_calls": tool_calls,
                "iteration_count": state["iteration_count"] + 1,
                "thoughts": thoughts,
                "done": False,
            }
        
        content = ""
        if hasattr(response, 'content'):
            if isinstance(response.content, str):
                content = response.content
            elif isinstance(response.content, list):
                for block in response.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content += block.get("text", "")
                    elif hasattr(block, 'text'):
                        content += block.text
        
        logger.info(f"[ReActAgent] Final answer: {len(content)} chars")
        
        thoughts.append({
            "iteration": state["iteration_count"],
            "phase": "final_answer",
            "answer_length": len(content),
        })
        
        return {
            "answer": content,
            "done": True,
            "thoughts": thoughts,
            "confidence": 0.85,
        }
        
    except Exception as e:
        logger.exception(f"[ReActAgent] Failed: {e}")
        
        thoughts.append({
            "iteration": state["iteration_count"],
            "phase": "error",
            "error": str(e),
        })
        
        return {
            "answer": f"处理请求时出现错误: {str(e)[:100]}",
            "done": True,
            "errors": [{"error": str(e), "iteration": state["iteration_count"]}],
            "thoughts": thoughts,
        }
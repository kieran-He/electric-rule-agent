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
1. retrieve_policy: 检索电力市场政策文档和法规（知识库）
2. fetch_electricity_data: 获取电力数据（负荷、发电量、电价等）
3. analyze_statistics: 对数据进行统计分析
4. web_search: 网络搜索，获取最新新闻、政策动态

数据可用性:
- fetch_electricity_data 目前只支持陕西省(SN)的数据查询
- 如果用户询问其他省份的数据，请告知当前只支持陕西数据

省份代码对照（重要）:
- 陕西 = SN, 山西 = SX, 山东 = SD
- 河南 = HA, 湖北 = HB, 湖南 = HN
- 江苏 = JS, 浙江 = ZJ, 安徽 = AH
- 北京 = BJ, 上海 = SH, 广东 = GD
- 其他省份请参考此格式使用两位大写字母代码

工具调用规则（重要）:
- retrieve_policy 和 web_search：只需要调用工具，无需提供query参数（系统会自动使用用户原始问题）
- provinces参数：如果问题明确提到省份，提供正确的省份代码（如["SN", "HA"]）；否则可省略，系统会自动推断
- fetch_electricity_data：需要提供 metric 参数（load/generation/price/new_energy）

工作流程:
1. 分析用户问题，确定需要使用哪些工具
2. 直接调用工具获取信息（无需自己生成查询内容）
3. 综合工具结果，生成完整回答

注意事项:
- 用户提到"最新"、"最近"、"新闻"等关键词 → 使用 web_search
- 政策法规问题 → 使用 retrieve_policy
- 数据查询 → 使用 fetch_electricity_data
- 数据分析 → 使用 analyze_statistics
- **重要**: 当获取到足够信息后，直接给出答案，不要继续调用工具
- **重要**: 如果系统提示"信息已充足"，必须直接生成答案，不要再调用任何工具"""


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
        
        # 重要：每轮都提醒LLM原始问题，防止偏离
        query = state["query"]
        provinces = state.get("provinces", ["SN"])
        province_str = ", ".join(provinces)
        
        # 检查信息充足度
        sufficient_info = state.get("sufficient_info", False)
        sufficiency_reason = state.get("sufficiency_reason", "")
        
        if sufficient_info:
            # 强制生成答案
            reminder = f"""

【系统提示】信息已充足: {sufficiency_reason}
【重要】请直接基于上述工具结果生成完整答案，不要再调用任何工具！
用户原始问题是: {query}"""
        else:
            reminder = f"""

【重要提醒】用户原始问题是: {query}
请基于上述工具结果，针对原始问题进行分析。如果结果不足以回答原始问题，可以继续调用工具；如果已经足够，请直接生成答案。"""
        messages.append(HumanMessage(content=reminder))
    
    # 首轮添加完整用户问题
    
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
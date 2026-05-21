import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple

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

工具调用规则（重要）:
- retrieve_policy 和 web_search：只需要调用工具，无需提供query参数（系统会自动使用用户原始问题）
- provinces参数：如果问题明确提到省份，提供省份代码（如["SN", "SD"]）；否则可省略，系统会自动推断
- fetch_electricity_data：需要提供 metric 参数（load/generation/price/new_energy）

工作流程:
1. 分析用户问题，确定需要使用哪些工具
2. 直接调用工具获取信息（无需自己生成查询内容）
3. 综合工具结果，生成完整回答

引用规范（必须遵守）:
- 当 retrieve_policy 返回参考内容时，每个chunk已标注编号如 `[chunk-1] 《文档名》`
- 在回答正文中引用时，使用格式：`[引用](#chunk-N)`
- 例如：根据规定，签约比例不低于80% [引用](#chunk-1)
- **禁止**使用其他格式如 `【chunk-N】` 或 `[chunk-N]` 或 `[《文档名》](#chunk-N)`
- 引用放在句末，只引用必要的来源
- **不要**在回答末尾添加"参考文件"或"参考文献"列表

注意事项:
- 用户提到"最新"、"最近"、"新闻"等关键词 → 使用 web_search
- 政策法规问题 → 使用 retrieve_policy
- 数据查询 → 使用 fetch_electricity_data
- 数据分析 → 使用 analyze_statistics
- **重要**: 当获取到足够信息后，直接给出答案，不要继续调用工具
- **重要**: 如果系统提示"信息已充足"，必须直接生成答案，不要再调用任何工具"""


def _fill_and_renumber_citations(answer: str, policy_chunks: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    后处理 LLM 答案中的引用：
    1. 将 [引用](#chunk-N) 替换为 [《docname》](#chunk-N)
    2. 提取答案中实际出现的 chunk 编号
    3. 重新编号为 1, 2, 3...
    4. 按出现顺序排序 chunks
    5. 移除未引用的 chunks
    """
    pattern = r'\[引用\]\(#chunk-(\d+)\)'
    matches = re.findall(pattern, answer)
    
    alt_pattern = r'【chunk-(\d+)】'
    alt_matches = re.findall(alt_pattern, answer)
    
    # 匹配 [《doc_name》](#chunk-N) 或 [doc_name](#chunk-N) 格式
    doc_pattern = r'\[《[^》]+》\]\(#chunk-(\d+)\)'
    doc_matches = re.findall(doc_pattern, answer)
    plain_pattern = r'\[[^\]]+\]\(#chunk-(\d+)\)'
    plain_matches = re.findall(plain_pattern, answer)
    
    all_matches = matches + alt_matches + doc_matches + plain_matches
    
    seen = set()
    cited_indices = []
    for m in all_matches:
        idx = int(m)
        if idx not in seen and idx <= len(policy_chunks):
            seen.add(idx)
            cited_indices.append(idx)
    
    renumber_map = {}
    for new_idx, old_idx in enumerate(cited_indices, 1):
        renumber_map[old_idx] = new_idx
    
    processed_answer = answer
    for old_idx, new_idx in renumber_map.items():
        chunk = policy_chunks[old_idx - 1]
        doc_name = chunk.get("source", "未知文档")
        # 替换标准格式 [引用](#chunk-N) -> [doc_name](#chunk-M)
        processed_answer = processed_answer.replace(
            f'[引用](#chunk-{old_idx})',
            f'[{doc_name}](#chunk-{new_idx})'
        )
        # 替换备用格式 【chunk-N】 -> [doc_name](#chunk-M)
        processed_answer = processed_answer.replace(
            f'【chunk-{old_idx}】',
            f'[{doc_name}](#chunk-{new_idx})'
        )
        # 重新编号已有格式 [《xxx》](#chunk-N) 或 [xxx](#chunk-N) -> [doc_name](#chunk-M)
        processed_answer = re.sub(
            rf'\[《[^》]+》\]\(#chunk-{old_idx}\)',
            f'[{doc_name}](#chunk-{new_idx})',
            processed_answer
        )
        processed_answer = re.sub(
            rf'\[[^\]]+\]\(#chunk-{old_idx}\)',
            f'[{doc_name}](#chunk-{new_idx})',
            processed_answer
        )
    
    # 清理多余的右括号
    processed_answer = re.sub(r'\]\(#chunk-\d+\)\]', '](#chunk-X)', processed_answer)
    processed_answer = processed_answer.replace('](#chunk-X)', '](#chunk-1)')
    
    # 清理所有未处理的引用格式残留
    processed_answer = re.sub(r'\[引用\]\(#chunk-\d+\)', '', processed_answer)
    processed_answer = re.sub(r'【chunk-\d+】', '', processed_answer)
    
    # 删除末尾的"参考文件"部分（更精确匹配）
    processed_answer = re.sub(
        r'\n---+\n[\*\*]*(参考文件|参考文献)[\*\*]*[：:].*',
        '',
        processed_answer,
        flags=re.DOTALL
    )
    
    ordered_chunks = [policy_chunks[old_idx - 1] for old_idx in cited_indices]
    
    logger.info(f"[ReActAgent] Citation post-processing: {len(all_matches)} citations, {len(ordered_chunks)} chunks kept")
    
    return processed_answer, ordered_chunks


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
        need_web_search = state.get("need_web_search", False)
        retrieval_quality = state.get("retrieval_quality", {})
        
        # 检查是否需要 web_search 提示
        has_web_search = any(r.get("tool_name") == "web_search" for r in state.get("tool_results", []))
        web_hint = ""
        if need_web_search and not has_web_search:
            reason = retrieval_quality.get("reason", "结果不足")
            web_hint = f"""

【检索质量提示】知识库检索结果质量较低: {reason}
建议调用 web_search 工具补充信息。"""
        
        if sufficient_info:
            # 强制生成答案
            reminder = f"""
{web_hint}
【系统提示】信息已充足: {sufficiency_reason}
【重要】请直接基于上述工具结果生成完整答案，不要再调用任何工具！
用户原始问题是: {query}"""
        else:
            reminder = f"""
{web_hint}
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
        
        # 后处理引用
        policy_chunks = state.get("policy_chunks", [])
        ordered_chunks = policy_chunks
        if policy_chunks:
            content, ordered_chunks = _fill_and_renumber_citations(content, policy_chunks)
        
        logger.info(f"[ReActAgent] Final answer: {len(content)} chars, {len(ordered_chunks)} chunks")
        
        thoughts.append({
            "iteration": state["iteration_count"],
            "phase": "final_answer",
            "answer_length": len(content),
            "chunks_kept": len(ordered_chunks),
        })
        
        return {
            "answer": content,
            "policy_chunks": ordered_chunks,
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
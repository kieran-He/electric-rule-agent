from __future__ import annotations

import logging
from typing import Dict, List, TYPE_CHECKING

from app.agent.graph.state import ElectricityAgentState

if TYPE_CHECKING:
    from app.langchain.llm import MiniMaxLLMWrapper

logger = logging.getLogger(__name__)


def format_policy_chunks(chunks: List[Dict]) -> str:
    if not chunks:
        return "未找到相关政策文档。"
    
    parts = []
    for i, chunk in enumerate(chunks[:5]):
        content = chunk.get("content", "")
        source = chunk.get("source", "未知来源")
        parts.append(f"[{i+1}] {source}: {content[:500]}")
    
    return "\n\n".join(parts)


def format_electricity_data(data: Dict) -> str:
    if not data or data.get("error"):
        return "未能获取电力数据。"
    
    metric = data.get("metric", "未知指标")
    province = data.get("province", "未知省份")
    data_points = data.get("data_points", 0)
    values = data.get("data", [])
    
    if not values:
        return f"{province}省份{metric}数据暂无。"
    
    return f"{province}省份{metric}数据：共{data_points}个数据点，最近值: {values[-1] if values else '无'}"


def format_analysis_result(result: Dict) -> str:
    if not result or result.get("error"):
        return "未能完成数据分析。"
    
    parts = []
    for key, value in result.items():
        if isinstance(value, dict):
            parts.append(f"{key}: {value}")
        else:
            parts.append(f"{key}: {value}")
    
    return "统计分析结果：" + ", ".join(parts)


def calculate_confidence(state: ElectricityAgentState) -> float:
    tool_calls = state.get("tool_calls", [])
    policy_chunks = state.get("policy_chunks", [])
    electricity_data = state.get("electricity_data")
    analysis_result = state.get("analysis_result")
    
    confidence = 0.3
    
    if policy_chunks and len(policy_chunks) > 0:
        confidence += 0.2
    
    if electricity_data and not electricity_data.get("error"):
        confidence += 0.2
    
    if analysis_result and not analysis_result.get("error"):
        confidence += 0.2
    
    if len(tool_calls) > 1:
        confidence += 0.1
    
    return min(confidence, 1.0)


def response_generator_node(state: ElectricityAgentState) -> Dict:
    query = state["query"]
    intent = state["intent"]
    policy_chunks = state.get("policy_chunks", [])
    electricity_data = state.get("electricity_data")
    analysis_result = state.get("analysis_result")
    
    from app.agent.graph.electricity_agent_graph import _get_current_instance
    graph_instance = _get_current_instance()
    
    llm_wrapper = None
    if graph_instance:
        llm_wrapper = graph_instance.llm_wrapper
    
    context_parts = []
    
    if policy_chunks:
        context_parts.append(format_policy_chunks(policy_chunks))
    
    if electricity_data:
        context_parts.append(format_electricity_data(electricity_data))
    
    if analysis_result:
        context_parts.append(format_analysis_result(analysis_result))
    
    context_str = "\n\n".join(context_parts) if context_parts else "暂无相关信息。"
    
    system_prompt = f"""你是一个电力政策与数据分析助手。根据以下信息回答用户问题。

用户意图: {intent}

相关信息:
{context_str}

请根据以上信息，简洁、准确地回答用户问题。如果信息不足，请说明。"""

    answer = ""
    if llm_wrapper:
        try:
            answer = llm_wrapper.invoke_text(query, system=system_prompt)
            logger.info(f"[ResponseGenerator] LLM generated answer: {len(answer)} chars")
        except Exception as e:
            logger.exception(f"[ResponseGenerator] LLM failed: {e}")
            answer = f"根据分析结果：{context_str}"
    else:
        answer = f"根据分析结果：{context_str}"
    
    confidence = calculate_confidence(state)
    
    return {
        "answer": answer,
        "confidence": confidence,
    }
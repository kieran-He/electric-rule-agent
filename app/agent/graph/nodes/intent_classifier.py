from __future__ import annotations

import logging
from typing import Dict

from app.agent.graph.state import ElectricityAgentState

logger = logging.getLogger(__name__)


def intent_classifier_node(state: ElectricityAgentState) -> Dict:
    query = state["query"]
    
    policy_keywords = ["政策", "规则", "通知", "规定", "条款", "准入", "交易规则"]
    data_keywords = ["负荷", "发电量", "用电量", "电价", "实时", "曲线", "数据"]
    analysis_keywords = ["统计", "均值", "方差", "分析", "趋势", "增长", "分布"]
    
    has_policy = any(kw in query for kw in policy_keywords)
    has_data = any(kw in query for kw in data_keywords)
    has_analysis = any(kw in query for kw in analysis_keywords)
    
    if has_policy and (has_data or has_analysis):
        intent = "hybrid"
    elif has_analysis:
        intent = "analysis"
    elif has_data:
        intent = "data_query"
    elif has_policy:
        intent = "policy_query"
    else:
        intent = "hybrid"
    
    logger.info(f"[IntentClassifier] Query: {query[:50]}, Intent: {intent}")
    
    return {
        "intent": intent,
    }
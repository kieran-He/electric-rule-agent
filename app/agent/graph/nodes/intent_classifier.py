from __future__ import annotations

import logging
from typing import Dict, TYPE_CHECKING

from app.agent.graph.state import ElectricityAgentState
from app.agent.graph.nodes.intent_rules import RuleClassifier
from app.agent.graph.nodes.intent_llm import LLMClassifier

if TYPE_CHECKING:
    from app.langchain.llm import MiniMaxLLMWrapper

logger = logging.getLogger(__name__)

RULE_THRESHOLD = 0.85
LLM_THRESHOLD = 0.70


def intent_classifier_node(state: ElectricityAgentState) -> Dict:
    query = state["query"]
    
    rule_result = RuleClassifier.classify(query)
    logger.info(f"[IntentClassifier] Rule result: intent={rule_result['intent']}, confidence={rule_result['confidence']:.2f}")
    
    if rule_result["confidence"] >= RULE_THRESHOLD:
        logger.info(f"[IntentClassifier] High confidence rule result, skipping LLM")
        return {
            "intent": rule_result["intent"],
            "intent_confidence": rule_result["confidence"],
            "intent_reason": rule_result["reason"],
            "sub_intents": rule_result.get("sub_intents", []),
            "provinces": rule_result.get("detected_regions", []) or state.get("provinces", []),
        }
    
    from app.agent.graph.electricity_agent_graph import _get_current_instance
    graph_instance = _get_current_instance()
    
    llm_wrapper = None
    if graph_instance:
        llm_wrapper = graph_instance.llm_wrapper
    
    llm_result = LLMClassifier.classify(query, rule_result, llm_wrapper)
    logger.info(f"[IntentClassifier] LLM result: intent={llm_result['intent']}, confidence={llm_result['confidence']:.2f}")
    
    provinces = state.get("provinces", [])
    detected_regions = llm_result.get("detected_regions", []) or rule_result.get("detected_regions", [])
    if detected_regions:
        provinces = list(set(provinces + detected_regions))
    
    return {
        "intent": llm_result["intent"],
        "intent_confidence": llm_result["confidence"],
        "intent_reason": llm_result["reason"],
        "sub_intents": llm_result.get("sub_intents", []),
        "provinces": provinces,
    }
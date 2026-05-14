from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Any, TYPE_CHECKING

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph.state import ElectricityAgentState
from app.agent.graph.nodes import (
    intent_classifier_node,
    policy_retriever_node,
    data_fetcher_node,
    data_analyzer_node,
    response_generator_node,
)
from app.schemas.agent import AgentRequest, AgentResponse

if TYPE_CHECKING:
    from app.langchain.orchestrator_hybrid import HybridQAOrchestrator
    from app.langchain.llm import MiniMaxLLMWrapper
    from app.agent.adapters.electricity_data_adapter import ElectricityDataAdapter
    from app.config import Settings

logger = logging.getLogger(__name__)

_current_instance: Optional["ElectricityAgentGraph"] = None


def _get_current_instance() -> Optional["ElectricityAgentGraph"]:
    return _current_instance


class ElectricityAgentGraph:
    def __init__(
        self,
        llm_wrapper: "MiniMaxLLMWrapper",
        orchestrator: "HybridQAOrchestrator",
        data_adapter: "ElectricityDataAdapter",
        settings: "Settings",
        checkpointer=None,
    ):
        global _current_instance
        self.llm_wrapper = llm_wrapper
        self.orchestrator = orchestrator
        self.data_adapter = data_adapter
        self.settings = settings
        
        self._graph = self._build_graph()
        self.checkpointer = checkpointer or MemorySaver()
        self.app = self._graph.compile(checkpointer=self.checkpointer)
        
        _current_instance = self
        
        logger.info("ElectricityAgentGraph initialized")
    
    @staticmethod
    def _get_current_instance():
        return _current_instance

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ElectricityAgentState)
        
        workflow.add_node("intent_classifier", intent_classifier_node)
        workflow.add_node("policy_retriever", policy_retriever_node)
        workflow.add_node("data_fetcher", data_fetcher_node)
        workflow.add_node("data_analyzer", data_analyzer_node)
        workflow.add_node("response_generator", response_generator_node)
        
        workflow.set_entry_point("intent_classifier")
        
        # Intent classifier routes to appropriate starting node
        workflow.add_conditional_edges(
            "intent_classifier",
            self._route_by_intent,
            {
                "policy": "policy_retriever",
                "data": "data_fetcher",
                "analysis": "data_fetcher",
                "hybrid": "policy_retriever",  # Start with policy, then get data
                "end": END,
            }
        )
        
        # Data path: fetcher -> analyzer -> response
        workflow.add_edge("data_fetcher", "data_analyzer")
        workflow.add_edge("data_analyzer", "response_generator")
        
        # Policy path: policy_query goes directly to response
        # Hybrid path: policy -> data_fetcher -> data_analyzer -> response
        workflow.add_conditional_edges(
            "policy_retriever",
            self._route_after_policy,
            {
                "generate": "response_generator",
                "fetch_data": "data_fetcher",
            }
        )
        
        workflow.add_edge("response_generator", END)
        
        return workflow
    
    def _route_after_policy(self, state: ElectricityAgentState) -> str:
        """After policy retrieval, decide if we also need data."""
        intent = state.get("intent", "hybrid")
        
        # For hybrid intent, also fetch data after getting policy
        if intent == "hybrid":
            return "fetch_data"
        
        # For pure policy queries, go directly to response generation
        return "generate"

    def _route_by_intent(self, state: ElectricityAgentState) -> str:
        intent = state.get("intent", "hybrid")
        
        if intent == "policy_query":
            return "policy"
        elif intent == "data_query":
            return "data"
        elif intent == "analysis":
            return "analysis"
        elif intent == "hybrid":
            return "hybrid"
        else:
            return "end"

    def run(
        self,
        query: str,
        provinces: List[str] = None,
        session_id: str = None,
        history: List[Dict] = None,
        context: Dict[str, Any] = None,
    ) -> Dict:
        initial_state: ElectricityAgentState = {
            "query": query,
            "provinces": provinces or ["SN"],
            "messages": history or [],
            "metadata": {
                "session_id": session_id,
                "context": context or {},
            },
            "intent": "",
            "intent_confidence": 0.0,
            "intent_reason": "",
            "sub_intents": [],
            "policy_chunks": [],
            "electricity_data": None,
            "analysis_result": None,
            "answer": "",
            "tool_calls": [],
            "confidence": 0.0,
        }
        
        config = {
            "configurable": {
                "thread_id": session_id or "default",
            }
        }
        
        result = self.app.invoke(initial_state, config)
        
        return {
            "answer": result.get("answer", ""),
            "intent": result.get("intent", ""),
            "tool_calls": result.get("tool_calls", []),
            "confidence": result.get("confidence", 0.0),
            "metadata": result.get("metadata", {}),
        }

    def stream(
        self,
        query: str,
        provinces: List[str] = None,
        session_id: str = None,
    ):
        initial_state: ElectricityAgentState = {
            "query": query,
            "provinces": provinces or ["SN"],
            "messages": [],
            "metadata": {
                "session_id": session_id,
            },
            "intent": "",
            "intent_confidence": 0.0,
            "intent_reason": "",
            "sub_intents": [],
            "policy_chunks": [],
            "electricity_data": None,
            "analysis_result": None,
            "answer": "",
            "tool_calls": [],
            "confidence": 0.0,
        }
        
        config = {
            "configurable": {
                "thread_id": session_id or "default",
            }
        }
        
        for event in self.app.stream(initial_state, config):
            yield event
    
    def chat(
        self,
        request: AgentRequest,
        db: Any = None,
        trace_service: Any = None,
    ) -> AgentResponse:
        trace_id = f"agent_{uuid.uuid4().hex[:12]}"
        
        result = self.run(
            query=request.query,
            provinces=request.province_codes,
            session_id=request.session_id,
            history=request.history,
            context=request.context,
        )
        
        return AgentResponse(
            answer=result.get("answer", ""),
            intent=result.get("intent", ""),
            tool_calls=result.get("tool_calls", []),
            citations=[],
            metadata=result.get("metadata", {}),
            confidence=result.get("confidence", 0.0),
            trace_id=trace_id,
            detected_provinces=request.province_codes[0] if request.province_codes else None,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "framework": "langgraph",
            "nodes": ["intent_classifier", "policy_retriever", "data_fetcher", "data_analyzer", "response_generator"],
            "data_adapter": type(self.data_adapter).__name__,
        }
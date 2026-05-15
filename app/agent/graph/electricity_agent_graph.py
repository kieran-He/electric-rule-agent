from __future__ import annotations

import logging
import uuid
import time
from typing import Dict, List, Any, TYPE_CHECKING

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph.state import ElectricityAgentState, create_initial_state
from app.agent.graph.nodes import (
    intent_classifier_node,
    policy_retriever_node,
    data_fetcher_node,
    data_analyzer_node,
    response_generator_node,
)
from app.agent.graph.nodes.react_agent_node import react_agent_node
from app.agent.graph.nodes.tool_executor_node import tool_executor_node
from app.agent.graph.handlers.iteration_control import IterationController
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


def _should_continue(state: ElectricityAgentState) -> str:
    """Route decision for ReAct loop with loop detection."""
    graph_instance = _get_current_instance()
    
    if state.get("done", False):
        return "end"
    
    if graph_instance and graph_instance._iteration_controller:
        metadata = state.get("metadata", {})
        start_time = metadata.get("start_time", time.time())
        elapsed = time.time() - start_time
        
        should_continue, reason = graph_instance._iteration_controller.should_continue(
            state, elapsed
        )
        
        if not should_continue:
            logger.warning(f"[ElectricityAgent] Loop stopped: {reason}")
            return "end"
    
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)
    
    if iteration_count >= max_iterations:
        logger.warning(f"[ElectricityAgent] Max iterations reached: {iteration_count}")
        return "end"
    
    if state.get("tool_calls"):
        return "tools"
    
    return "end"


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
        
        self.use_react = getattr(settings, 'agent_use_react', True)
        self.max_iterations = getattr(settings, 'agent_max_iterations', 5)
        self.tool_timeout = getattr(settings, 'agent_tool_timeout', 30)
        
        self._iteration_controller = IterationController(
            max_iterations=self.max_iterations,
            timeout_seconds=self.tool_timeout,
        )
        
        self._graph = self._build_graph()
        self.checkpointer = checkpointer or MemorySaver()
        self.app = self._graph.compile(checkpointer=self.checkpointer)
        
        _current_instance = self
        
        logger.info(f"ElectricityAgentGraph initialized (react={self.use_react}, max_iter={self.max_iterations})")
    
    @staticmethod
    def _get_current_instance():
        return _current_instance
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ElectricityAgentState)
        
        if self.use_react:
            workflow.add_node("react_agent", react_agent_node)
            workflow.add_node("tool_executor", tool_executor_node)
            
            workflow.set_entry_point("react_agent")
            
            workflow.add_conditional_edges(
                "react_agent",
                _should_continue,
                {
                    "tools": "tool_executor",
                    "end": END,
                }
            )
            workflow.add_edge("tool_executor", "react_agent")
            
            logger.info("[ElectricityAgent] ReAct graph built")
        else:
            workflow.add_node("intent_classifier", intent_classifier_node)
            workflow.add_node("policy_retriever", policy_retriever_node)
            workflow.add_node("data_fetcher", data_fetcher_node)
            workflow.add_node("data_analyzer", data_analyzer_node)
            workflow.add_node("response_generator", response_generator_node)
            
            workflow.set_entry_point("intent_classifier")
            
            workflow.add_conditional_edges(
                "intent_classifier",
                self._route_by_intent,
                {
                    "policy": "policy_retriever",
                    "data": "data_fetcher",
                    "analysis": "data_fetcher",
                    "hybrid": "policy_retriever",
                    "end": END,
                }
            )
            
            workflow.add_edge("data_fetcher", "data_analyzer")
            workflow.add_edge("data_analyzer", "response_generator")
            
            workflow.add_conditional_edges(
                "policy_retriever",
                self._route_after_policy,
                {
                    "generate": "response_generator",
                    "fetch_data": "data_fetcher",
                }
            )
            
            workflow.add_edge("response_generator", END)
            
            logger.info("[ElectricityAgent] Fixed routing graph built")
        
        return workflow
    
    def _route_after_policy(self, state: ElectricityAgentState) -> str:
        intent = state.get("intent", "hybrid")
        if intent == "hybrid":
            return "fetch_data"
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
        initial_state = create_initial_state(
            query=query,
            provinces=provinces,
            session_id=session_id,
            history=history,
            context=context,
            max_iterations=self.max_iterations,
        )
        
        config = {
            "configurable": {
                "thread_id": session_id or "default",
            }
        }
        
        start_time = time.time()
        result = self.app.invoke(initial_state, config)
        elapsed = time.time() - start_time
        
        logger.info(f"[ElectricityAgent] Run completed in {elapsed:.2f}s, iterations={result.get('iteration_count', 0)}")
        
        chart_paths = result.get("chart_paths", [])
        if chart_paths:
            logger.info(f"[ElectricityAgent] Found {len(chart_paths)} chart paths: {chart_paths}")
        
        return {
            "answer": result.get("answer", ""),
            "intent": result.get("intent", ""),
            "tool_calls": [r.get("tool_name") for r in result.get("tool_results", [])],
            "confidence": result.get("confidence", 0.0),
            "metadata": {
                **result.get("metadata", {}),
                "elapsed_seconds": elapsed,
                "iterations": result.get("iteration_count", 0),
                "thoughts": result.get("thoughts", []),
            },
            "policy_chunks": result.get("policy_chunks", []),
            "electricity_data": result.get("electricity_data"),
            "chart_paths": result.get("chart_paths", []),
        }

    def stream(
        self,
        query: str,
        provinces: List[str] = None,
        session_id: str = None,
    ):
        initial_state = create_initial_state(
            query=query,
            provinces=provinces,
            session_id=session_id,
            max_iterations=self.max_iterations,
        )
        
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
            chart_paths=result.get("chart_paths", []),
        )
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "framework": "langgraph",
            "react_mode": self.use_react,
            "max_iterations": self.max_iterations,
            "nodes": ["react_agent", "tool_executor"] if self.use_react else ["intent_classifier", "policy_retriever", "data_fetcher", "data_analyzer", "response_generator"],
            "data_adapter": type(self.data_adapter).__name__,
        }
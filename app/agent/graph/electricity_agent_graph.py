from __future__ import annotations

import contextvars
import logging
import uuid
import time
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph.state import ElectricityAgentState, create_initial_state
from app.agent.graph.nodes.react_agent_node import react_agent_node
from app.agent.graph.nodes.tool_executor_node import tool_executor_node
from app.agent.graph.handlers.iteration_control import IterationController
from app.schemas.agent import AgentRequest, AgentResponse

if TYPE_CHECKING:
    from app.langchain.orchestrator_hybrid import HybridQAOrchestrator
    from app.langchain.llm import MiniMaxLLMWrapper
    from app.agent.adapters.electricity_data_adapter import ElectricityDataAdapter
    from app.config import Settings
    from app.core.web_search import WebSearchClient

logger = logging.getLogger(__name__)

_current_instance: contextvars.ContextVar[Optional["ElectricityAgentGraph"]] = (
    contextvars.ContextVar("_current_instance", default=None)
)


def _get_current_instance() -> Optional["ElectricityAgentGraph"]:
    return _current_instance.get()


def _set_current_instance(instance: "ElectricityAgentGraph") -> None:
    _current_instance.set(instance)


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
        web_search_client: "WebSearchClient" = None,
        checkpointer=None,
    ):
        self.llm_wrapper = llm_wrapper
        self.orchestrator = orchestrator
        self.data_adapter = data_adapter
        self.settings = settings
        self.web_search_client = web_search_client
        
        self.max_iterations = getattr(settings, 'agent_max_iterations', 5)
        self.tool_timeout = getattr(settings, 'agent_tool_timeout', 30)
        
        self._iteration_controller = IterationController(
            max_iterations=self.max_iterations,
            timeout_seconds=self.tool_timeout,
        )
        
        self._graph = self._build_graph()
        self.checkpointer = checkpointer or MemorySaver()
        self.app = self._graph.compile(checkpointer=self.checkpointer)
        
        logger.info(f"ElectricityAgentGraph initialized (max_iter={self.max_iterations})")
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ElectricityAgentState)
        
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
        
        return workflow
    
    def run(
        self,
        query: str,
        provinces: List[str] = None,
        session_id: str = None,
        history: List[Dict] = None,
        context: Dict[str, Any] = None,
    ) -> Dict:
        _set_current_instance(self)
        
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
        
        answer = result.get("answer", "")
        if not answer and result.get("tool_results"):
            logger.warning("[ElectricityAgent] No final answer but has tool results, generating fallback response")
            answer = self._generate_fallback_answer(query, result.get("tool_results", []))
        
        chart_paths = result.get("chart_paths", [])
        if chart_paths:
            logger.info(f"[ElectricityAgent] Found {len(chart_paths)} chart paths: {chart_paths}")
        
        return {
            "answer": answer,
            "intent": result.get("intent", ""),
            "tool_calls": [r.get("tool_name") for r in result.get("tool_results", [])],
            "confidence": result.get("confidence", 0.0),
            "metadata": {
                **result.get("metadata", {}),
                "elapsed_seconds": elapsed,
                "iterations": result.get("iteration_count", 0),
                "thoughts": result.get("thoughts", []),
                "timeout": elapsed > self.tool_timeout,
            },
            "policy_chunks": result.get("policy_chunks", []),
            "electricity_data": result.get("electricity_data"),
            "chart_paths": result.get("chart_paths", []),
        }
    
    def _generate_fallback_answer(self, query: str, tool_results: List[Dict]) -> str:
        """Generate a fallback answer when agent times out without final answer."""
        policy_chunks = []
        web_results = []
        
        for result in tool_results:
            tool_name = result.get("tool_name", "")
            output = result.get("output", "")
            if tool_name == "retrieve_policy" and output:
                policy_chunks.append(output)
            elif tool_name == "web_search" and output:
                web_results.append(output)
        
        if policy_chunks or web_results:
            fallback = "抱歉，处理超时，但已获取以下相关信息：\n\n"
            
            if policy_chunks:
                fallback += "**政策检索结果：**\n"
                for chunk in policy_chunks[:2]:
                    fallback += chunk[:500] + "\n...\n"
            
            if web_results:
                fallback += "\n**网络搜索结果：**\n"
                for result in web_results[:1]:
                    fallback += result[:500] + "\n...\n"
            
            fallback += "\n请稍后重试或简化问题。"
            return fallback
        
        return "抱歉，处理超时，未能获取完整答案。请稍后重试或简化问题。"

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
        
        context = request.context or {}
        context["show_chunks"] = getattr(request, "show_chunks", True)
        
        result = self.run(
            query=request.query,
            provinces=request.province_codes,
            session_id=request.session_id,
            history=request.history,
            context=context,
        )
        
        citations = []
        policy_chunks = result.get("policy_chunks", [])
        if policy_chunks:
            from app.schemas.answer import CitationItem
            for chunk_data in policy_chunks[:8]:
                if isinstance(chunk_data, dict):
                    citation = CitationItem(
                        doc_name=chunk_data.get("source", ""),
                        status="formal",
                        title_path=chunk_data.get("title_path", ""),
                        excerpt=chunk_data.get("content", "")[:260],
                        issuer=chunk_data.get("issuer"),
                        issue_date=chunk_data.get("issue_date"),
                        effective_date=chunk_data.get("effective_date"),
                    )
                    citations.append(citation)
        
        return AgentResponse(
            answer=result.get("answer", ""),
            intent=result.get("intent", ""),
            tool_calls=result.get("tool_calls", []),
            citations=citations,
            metadata=result.get("metadata", {}),
            confidence=result.get("confidence", 0.0),
            trace_id=trace_id,
            detected_provinces=request.province_codes[0] if request.province_codes else None,
            chart_paths=result.get("chart_paths", []),
        )
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "framework": "langgraph",
            "max_iterations": self.max_iterations,
            "nodes": ["react_agent", "tool_executor"],
            "data_adapter": type(self.data_adapter).__name__,
        }
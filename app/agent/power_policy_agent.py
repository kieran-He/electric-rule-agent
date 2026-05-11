from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, TYPE_CHECKING

from app.agent.intent_router import IntentRouter, IntentType
from app.agent.react_agent import ReActAgent
from app.agent.tools.base import BaseTool, ToolResult
from app.agent.tools.rag_tool import RAGTool
from app.agent.tools.web_search_tool import WebSearchTool
from app.agent.tools.general_chat_tool import GeneralChatTool
from app.langchain.query_rewriter import QueryRewriter, RewriteResult
from app.schemas.agent import AgentRequest, AgentResponse

if TYPE_CHECKING:
    from app.langchain.orchestrator_hybrid import HybridQAOrchestrator
    from app.langchain.llm import MiniMaxLLMWrapper
    from app.core.web_search import WebSearchClient
    from app.config import Settings

logger = logging.getLogger(__name__)


class PowerPolicyAgent:
    """
    Agent for power policy QA with multi-tool support using ReAct pattern.
    
    Flow:
    1. Pre-process: Query Rewrite (统一在Agent层执行一次)
    2. ReAct循环: LLM决定工具调用
    3. 工具执行: 使用已rewrite的query，避免重复rewrite
    """
    
    def __init__(
        self,
        orchestrator: "HybridQAOrchestrator",
        llm_wrapper: "MiniMaxLLMWrapper",
        settings: "Settings",
        web_search_client: "WebSearchClient" = None,
        use_react: bool = True,
        max_iterations: int = 3,
    ):
        self._orchestrator = orchestrator
        self._llm_wrapper = llm_wrapper
        self._settings = settings
        self._web_search_client = web_search_client
        self._use_react = use_react
        
        self._tools: List[BaseTool] = []
        self._router = IntentRouter()
        self._react_agent: ReActAgent = None
        self._query_rewriter: QueryRewriter = None
        
        self._init_tools()
        self._init_rewriter()
        
        if use_react and getattr(self._llm_wrapper, 'api_key', None):
            self._react_agent = ReActAgent(
                llm_wrapper=llm_wrapper,
                tools=self._tools,
                max_iterations=max_iterations,
            )
        
        logger.info(f"PowerPolicyAgent initialized: use_react={use_react}, tools={len(self._tools)}")
    
    def _init_tools(self) -> None:
        self._tools.append(RAGTool(self._orchestrator))
        
        if self._web_search_client:
            self._tools.append(WebSearchTool(
                self._web_search_client,
                self._llm_wrapper,
                self._settings,
            ))
        
        self._tools.append(GeneralChatTool(self._llm_wrapper))
    
    def _init_rewriter(self) -> None:
        """Initialize QueryRewriter for pre-processing."""
        rewrite_enabled = getattr(self._settings, 'query_rewrite_enabled', False)
        if rewrite_enabled and self._llm_wrapper:
            self._query_rewriter = QueryRewriter(
                llm_wrapper=self._llm_wrapper,
                enabled=True,
                always_rewrite=True,
            )
            logger.info("QueryRewriter initialized for agent-level preprocessing")
    
    def chat(self, request: AgentRequest, db: Any = None, trace_service: Any = None) -> AgentResponse:
        trace_id = f"agent_{uuid.uuid4().hex[:12]}"
        
        rewrite_result = self._preprocess_query(request.query)
        
        context: Dict[str, Any] = {
            "session_id": request.session_id,
            "province_codes": request.province_codes,
            "history": request.history,
            "top_k": getattr(request.context, "top_k", 8) if request.context else 8,
            "need_citation": getattr(request.context, "need_citation", True) if request.context else True,
            "db": db,
            "trace_service": trace_service,
            "orchestrator": self._orchestrator,
            "rewrite_result": rewrite_result,
            "rewrite_done": True,
        }
        
        if self._react_agent:
            return self._run_react(request, context, trace_id)
        
        return self._run_single_tool(request, context, trace_id)
    
    def _preprocess_query(self, query: str) -> RewriteResult:
        """
        Pre-process query with rewriting at Agent level.
        
        This ensures query rewrite happens only once,
        avoiding repeated rewrite in HybridRetriever.
        """
        if self._query_rewriter:
            result = self._query_rewriter.rewrite(query)
            if result.triggered:
                logger.info(f"[Agent] Query pre-processed: {query} -> {len(result.queries)} queries")
                return result
        
        from app.langchain.query_rewriter import QueryPlan
        from dataprocess.province_mapping import PROVINCE_ALIASES
        
        codes = []
        for alias, code in PROVINCE_ALIASES.items():
            if alias in query:
                codes.append(code)
        
        return RewriteResult(
            queries=[QueryPlan(query, codes)],
            should_split=False,
            split_reason="no_rewriter",
            triggered=False,
            trigger_reason="disabled"
        )
    
    def _run_react(self, request: AgentRequest, context: Dict[str, Any], trace_id: str) -> AgentResponse:
        """Execute query using ReAct agent for multi-tool orchestration."""
        logger.info(f"[ReAct] Starting execution for query: {request.query[:50]}...")
        
        for tool in self._tools:
            tool.set_context(context)
        
        result = self._react_agent.run(request.query, context)
        
        output = result.get("output", "")
        tool_calls = result.get("tool_calls", [])
        
        success = result.get("success", False)
        
        if "知识库中未找到" in output and tool_calls == ["rag"]:
            intent = "no_result"
        elif tool_calls:
            intent = "multi_tool" if len(tool_calls) > 1 else tool_calls[0]
        else:
            intent = "general_chat"
        
        logger.info(f"[ReAct] Completed: tool_calls={tool_calls}, intent={intent}")
        
        return AgentResponse(
            answer=output,
            intent=intent,
            tool_calls=tool_calls,
            citations=[],
            metadata={"react_success": success},
            confidence=0.7 if success else 0.3,
            trace_id=trace_id,
            detected_provinces=request.province_codes[0] if request.province_codes else None,
        )
    
    def _run_single_tool(self, request: AgentRequest, context: Dict[str, Any], trace_id: str) -> AgentResponse:
        """Fallback single-tool routing using IntentRouter."""
        tool = self._router.route(request.query, self._tools)
        intent = self._router.detect_intent(request.query, tool)
        
        logger.info(f"Intent routed to {tool.name} for query: {request.query[:50]}...")
        
        result: ToolResult = tool.execute(request.query, context)
        
        return AgentResponse(
            answer=result.output,
            intent=intent.value,
            tool_calls=[tool.name],
            citations=result.citations,
            metadata=result.metadata,
            confidence=result.confidence,
            trace_id=trace_id,
            detected_provinces=result.metadata.get("detected_provinces"),
        )
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "tools": [t.name for t in self._tools],
            "use_react": self._use_react,
            "router": "react" if self._react_agent else "keyword_match",
            "max_iterations": self._react_agent._max_iterations if self._react_agent else None,
            "query_rewriter": self._query_rewriter is not None,
        }
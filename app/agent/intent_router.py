from __future__ import annotations

import logging
from enum import Enum
from typing import List, TYPE_CHECKING

from app.agent.tools.base import BaseTool

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class IntentType(Enum):
    POLICY_QA = "policy_qa"
    WEB_SEARCH = "web_search"
    GENERAL_CHAT = "general_chat"


class IntentRouter:
    """
    Intent router for tool selection.
    
    Routing strategy:
    1. Keyword fast matching - check each tool's keywords
    2. Priority: RAGTool (power domain) > WebSearchTool (latest info) > GeneralChatTool (fallback)
    """
    
    def __init__(self):
        self._tool_priority = {
            IntentType.POLICY_QA: "rag",
            IntentType.WEB_SEARCH: "web_search",
            IntentType.GENERAL_CHAT: "general_chat",
        }
    
    def route(self, query: str, tools: List[BaseTool]) -> BaseTool:
        rag_tool = None
        web_search_tool = None
        general_chat_tool = None
        
        for tool in tools:
            if tool.name == "rag":
                rag_tool = tool
            elif tool.name == "web_search":
                web_search_tool = tool
            elif tool.name == "general_chat":
                general_chat_tool = tool
        
        if rag_tool and rag_tool.is_applicable(query):
            logger.info(f"Intent routed to RAGTool for query: {query[:50]}")
            return rag_tool
        
        if web_search_tool and web_search_tool.is_applicable(query):
            logger.info(f"Intent routed to WebSearchTool for query: {query[:50]}")
            return web_search_tool
        
        if rag_tool:
            logger.info(f"Intent defaulted to RAGTool for query: {query[:50]}")
            return rag_tool
        
        if general_chat_tool:
            logger.info(f"Intent routed to GeneralChatTool (fallback) for query: {query[:50]}")
            return general_chat_tool
        
        raise RuntimeError("No tools available for routing")
    
    def detect_intent(self, query: str, selected_tool: BaseTool) -> IntentType:
        intent_map = {
            "rag": IntentType.POLICY_QA,
            "web_search": IntentType.WEB_SEARCH,
            "general_chat": IntentType.GENERAL_CHAT,
        }
        return intent_map.get(selected_tool.name, IntentType.GENERAL_CHAT)
from __future__ import annotations

import logging
import threading
from typing import Any, Optional, TYPE_CHECKING

from sqlalchemy.orm import Session

from app.agent.graph.electricity_agent_graph import ElectricityAgentGraph
from app.agent.adapters.electricity_data_adapter import create_data_adapter
from app.agent.graph.checkpointer.db_checkpointer import DbCheckpointer
from app.db.models.langgraph_checkpoint import LangGraphCheckpoint
from app.langchain.llm import MiniMaxLLMWrapper
from app.core.web_search import create_web_search_client
from app.services.orchestrator_singleton import orchestrator_singleton
from app.db.session import SessionLocal

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class AgentSingleton:
    _instance: Optional[AgentSingleton] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> AgentSingleton:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._agent: Optional[ElectricityAgentGraph] = None
                    cls._instance._settings: Optional["Settings"] = None
        return cls._instance
    
    def preload(self, settings: "Settings") -> ElectricityAgentGraph:
        if self._agent is not None:
            logger.debug("Agent already preloaded")
            return self._agent
        
        with self._lock:
            if self._agent is None:
                logger.info("Preloading Agent components...")
                
                orchestrator = orchestrator_singleton.preload(settings)
                
                import os
                llm_wrapper = MiniMaxLLMWrapper(
                    api_key=os.getenv("LLM_API_KEY", ""),
                    endpoint=os.getenv("LLM_ENDPOINT", "https://api.minimaxi.com/anthropic"),
                    model=os.getenv("LLM_MODEL", "MiniMax-M2.7"),
                    disable_thinking=True,
                )
                
                data_adapter = create_data_adapter(settings)
                web_search_client = create_web_search_client(settings)
                checkpointer = DbCheckpointer(SessionLocal)
                
                try:
                    with SessionLocal() as db:
                        db.query(LangGraphCheckpoint).limit(1).first()
                    logger.info("DbCheckpointer database connection verified")
                except Exception as e:
                    logger.warning(f"DbCheckpointer connection test failed: {e}")
                
                self._agent = ElectricityAgentGraph(
                    llm_wrapper=llm_wrapper,
                    orchestrator=orchestrator,
                    data_adapter=data_adapter,
                    settings=settings,
                    web_search_client=web_search_client,
                    checkpointer=checkpointer,
                )
                logger.info(f"ElectricityAgentGraph preloaded (LangGraph framework with DbCheckpointer)")
                
                self._settings = settings
        
        return self._agent
    
    def get_agent(self, db: Session) -> ElectricityAgentGraph:
        if self._agent is None:
            raise RuntimeError("Agent not preloaded. Call preload() first.")
        
        if hasattr(self._agent, '_orchestrator'):
            self._agent._orchestrator.db = db
        return self._agent
    
    def is_loaded(self) -> bool:
        return self._agent is not None
    
    def get_stats(self) -> dict:
        if self._agent:
            if hasattr(self._agent, 'get_stats'):
                return self._agent.get_stats()
            return {"framework": "langgraph", "loaded": True}
        return {"loaded": False}
    
    def clear(self) -> None:
        with self._lock:
            self._agent = None
            self._settings = None


agent_singleton = AgentSingleton()


def preload_agent(settings: "Settings") -> bool:
    try:
        agent = agent_singleton.preload(settings)
        return agent is not None
    except Exception as e:
        logger.error(f"Failed to preload agent: {e}")
        return False
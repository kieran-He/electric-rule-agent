from __future__ import annotations

import logging
import threading
from typing import Any, Optional, TYPE_CHECKING, Union

from sqlalchemy.orm import Session

from app.agent.power_policy_agent import PowerPolicyAgent
from app.agent.graph.electricity_agent_graph import ElectricityAgentGraph
from app.agent.adapters.electricity_data_adapter import create_data_adapter
from app.langchain.llm import MiniMaxLLMWrapper
from app.core.web_search import create_web_search_client
from app.services.orchestrator_singleton import orchestrator_singleton

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
                    cls._instance._agent: Optional[Union[PowerPolicyAgent, ElectricityAgentGraph]] = None
                    cls._instance._settings: Optional["Settings"] = None
                    cls._instance._framework: str = "react"
        return cls._instance
    
    def preload(self, settings: "Settings") -> Union[PowerPolicyAgent, ElectricityAgentGraph]:
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
                
                framework = getattr(settings, 'agent_framework', 'langgraph')
                self._framework = framework
                
                if framework == "langgraph":
                    data_adapter = create_data_adapter(settings)
                    
                    self._agent = ElectricityAgentGraph(
                        llm_wrapper=llm_wrapper,
                        orchestrator=orchestrator,
                        data_adapter=data_adapter,
                        settings=settings,
                    )
                    logger.info(f"ElectricityAgentGraph preloaded (LangGraph framework)")
                else:
                    web_search_client = create_web_search_client(settings)
                    
                    self._agent = PowerPolicyAgent(
                        orchestrator=orchestrator,
                        llm_wrapper=llm_wrapper,
                        settings=settings,
                        web_search_client=web_search_client,
                        use_react=settings.agent_use_react,
                        max_iterations=settings.agent_max_iterations,
                    )
                    logger.info(f"PowerPolicyAgent preloaded (ReAct framework)")
                
                self._settings = settings
        
        return self._agent
    
    def get_agent(self, db: Session) -> Union[PowerPolicyAgent, ElectricityAgentGraph]:
        if self._agent is None:
            raise RuntimeError("Agent not preloaded. Call preload() first.")
        
        if hasattr(self._agent, '_orchestrator'):
            self._agent._orchestrator.db = db
        return self._agent
    
    def get_framework(self) -> str:
        return self._framework
    
    def is_langgraph(self) -> bool:
        return self._framework == "langgraph"
    
    def is_loaded(self) -> bool:
        """Check if agent is preloaded."""
        return self._agent is not None
    
    def get_stats(self) -> dict:
        if self._agent:
            if hasattr(self._agent, 'get_stats'):
                return self._agent.get_stats()
            return {"framework": self._framework, "loaded": True}
        return {"loaded": False}
    
    def clear(self) -> None:
        with self._lock:
            self._agent = None
            self._settings = None
            self._framework = "react"


agent_singleton = AgentSingleton()


def preload_agent(settings: "Settings") -> bool:
    try:
        agent = agent_singleton.preload(settings)
        return agent is not None
    except Exception as e:
        logger.error(f"Failed to preload agent: {e}")
        return False
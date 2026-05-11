from __future__ import annotations

import logging
import threading
from typing import Any, Optional, TYPE_CHECKING

from sqlalchemy.orm import Session

from app.agent.power_policy_agent import PowerPolicyAgent
from app.langchain.llm import MiniMaxLLMWrapper
from app.core.web_search import create_web_search_client
from app.services.orchestrator_singleton import orchestrator_singleton

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class AgentSingleton:
    """
    Singleton cache for PowerPolicyAgent preloading.
    
    Thread-safe singleton pattern similar to OrchestratorSingleton.
    """
    
    _instance: Optional[AgentSingleton] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> AgentSingleton:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._agent: Optional[PowerPolicyAgent] = None
                    cls._instance._settings: Optional["Settings"] = None
        return cls._instance
    
    def preload(self, settings: "Settings") -> PowerPolicyAgent:
        """
        Preload agent components (orchestrator, llm, tools).
        
        Args:
            settings: Application settings
            
        Returns:
            Initialized PowerPolicyAgent
        """
        if self._agent is not None:
            logger.debug("Agent already preloaded")
            return self._agent
        
        with self._lock:
            if self._agent is None:
                logger.info("Preloading PowerPolicyAgent components...")
                
                orchestrator = orchestrator_singleton.preload(settings)
                
                import os
                llm_wrapper = MiniMaxLLMWrapper(
                    api_key=os.getenv("LLM_API_KEY", ""),
                    endpoint=os.getenv("LLM_ENDPOINT", "https://api.minimaxi.com/anthropic"),
                    model=os.getenv("LLM_MODEL", "MiniMax-M2.7"),
                    disable_thinking=True,
                )
                
                web_search_client = create_web_search_client(settings)
                
                self._agent = PowerPolicyAgent(
                    orchestrator=orchestrator,
                    llm_wrapper=llm_wrapper,
                    settings=settings,
                    web_search_client=web_search_client,
                    use_react=settings.agent_use_react,
                    max_iterations=settings.agent_max_iterations,
                )
                self._settings = settings
                logger.info(f"PowerPolicyAgent preloaded: use_react={settings.agent_use_react}, max_iterations={settings.agent_max_iterations}")
        
        return self._agent
    
    def get_agent(self, db: Session) -> PowerPolicyAgent:
        """
        Get agent for request processing.
        
        Args:
            db: Database session for metrics (per-request)
            
        Returns:
            PowerPolicyAgent instance
            
        Raises:
            RuntimeError: If agent not preloaded
        """
        if self._agent is None:
            raise RuntimeError("Agent not preloaded. Call preload() first.")
        
        self._agent._orchestrator.db = db
        return self._agent
    
    def is_loaded(self) -> bool:
        """Check if agent is preloaded."""
        return self._agent is not None
    
    def get_stats(self) -> dict:
        """Get agent stats if loaded."""
        if self._agent:
            return self._agent.get_stats()
        return {"loaded": False}
    
    def clear(self) -> None:
        """Clear cached agent (for testing or memory management)."""
        with self._lock:
            self._agent = None
            self._settings = None


agent_singleton = AgentSingleton()


def preload_agent(settings: "Settings") -> bool:
    """
    Convenience function to preload agent at application startup.
    
    Args:
        settings: Application settings
        
    Returns:
        True if preloaded successfully, False otherwise
    """
    try:
        agent = agent_singleton.preload(settings)
        return agent is not None
    except Exception as e:
        logger.error(f"Failed to preload agent: {e}")
        return False
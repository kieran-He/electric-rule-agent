"""
Orchestrator Singleton for Preloading

Preloads orchestrator components at application startup to avoid first-request latency.
"""
from __future__ import annotations

import threading
from typing import Any, Optional, TYPE_CHECKING

from sqlalchemy.orm import Session

from app.langchain.orchestrator_hybrid import HybridQAOrchestrator
import logging

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class OrchestratorSingleton:
    """
    Singleton cache for HybridQAOrchestrator preloading.
    
    Thread-safe singleton pattern to ensure orchestrator is initialized only once.
    Preloading reduces first-request latency from ~20s to instant.
    
    Note: db session is passed per-request since sessions cannot be reused.
    """
    
    _instance: Optional[OrchestratorSingleton] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> OrchestratorSingleton:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._orchestrator: Optional[HybridQAOrchestrator] = None
                    cls._instance._settings: Optional[Any] = None
        return cls._instance
    
    def preload(self, settings: "Settings") -> HybridQAOrchestrator:
        """
        Preload orchestrator components (embedding, reranker, BM25).
        
        Args:
            settings: Application settings
            
        Returns:
            Initialized HybridQAOrchestrator (without db session)
        """
        if self._orchestrator is not None:
            logger.debug("Orchestrator already preloaded")
            return self._orchestrator
        
        with self._lock:
            if self._orchestrator is None:
                logger.info("Preloading HybridQAOrchestrator components...")
                self._orchestrator = HybridQAOrchestrator(
                    db=None,
                    settings=settings,
                )
                self._settings = settings
                logger.info("HybridQAOrchestrator preloaded successfully")
        
        return self._orchestrator
    
    def get_orchestrator(self, db: Session) -> HybridQAOrchestrator:
        """
        Get orchestrator for request processing.
        
        Args:
            db: Database session for metrics (per-request)
            
        Returns:
            HybridQAOrchestrator instance
            
        Raises:
            RuntimeError: If orchestrator not preloaded
        """
        if self._orchestrator is None:
            raise RuntimeError("Orchestrator not preloaded. Call preload() first.")
        
        self._orchestrator.db = db
        return self._orchestrator
    
    def is_loaded(self) -> bool:
        """Check if orchestrator is preloaded."""
        return self._orchestrator is not None
    
    def get_stats(self) -> dict:
        """Get orchestrator stats if loaded."""
        if self._orchestrator:
            return self._orchestrator.get_retrieval_stats()
        return {"loaded": False}
    
    def clear(self) -> None:
        """Clear cached orchestrator (for testing or memory management)."""
        with self._lock:
            self._orchestrator = None
            self._settings = None


orchestrator_singleton = OrchestratorSingleton()


def preload_orchestrator(settings: "Settings") -> bool:
    """
    Convenience function to preload orchestrator at application startup.
    
    Args:
        settings: Application settings
        
    Returns:
        True if preloaded successfully, False otherwise
    """
    try:
        orchestrator = orchestrator_singleton.preload(settings)
        return orchestrator is not None
    except Exception as e:
        logger.error(f"Failed to preload orchestrator: {e}")
        return False
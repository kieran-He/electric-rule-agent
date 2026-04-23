"""
Reranker Cache Singleton for Preloading

Preloads reranker model at application startup to avoid first-request latency.
"""
from __future__ import annotations

import threading
from typing import Optional

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    CrossEncoder = None

import logging

logger = logging.getLogger(__name__)


class RerankerCache:
    """
    Singleton cache for reranker model preloading.
    
    Thread-safe singleton pattern to ensure model is loaded only once.
    Preloading reduces first-request latency from ~10s to ~3s.
    """
    
    _instance: Optional[RerankerCache] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> RerankerCache:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._reranker: Optional[CrossEncoder] = None
                    cls._instance._model_name: Optional[str] = None
        return cls._instance
    
    def preload(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        max_length: int = 512,
    ) -> Optional[CrossEncoder]:
        """
        Preload reranker model.
        
        Args:
            model_name: HuggingFace model name
            max_length: Max sequence length
            
        Returns:
            Loaded CrossEncoder model or None if unavailable
        """
        if not RERANKER_AVAILABLE:
            logger.warning("sentence-transformers not installed, reranker unavailable")
            return None
        
        if self._reranker is not None:
            if self._model_name != model_name:
                logger.info(f"Reloading reranker: {self._model_name} -> {model_name}")
                self._reranker = None
            else:
                logger.debug(f"Reranker already loaded: {model_name}")
                return self._reranker
        
        with self._lock:
            if self._reranker is None:
                logger.info(f"Preloading reranker model: {model_name}")
                self._reranker = CrossEncoder(model_name, max_length=max_length)
                self._model_name = model_name
                logger.info(f"Reranker model preloaded successfully")
        
        return self._reranker
    
    def get(self) -> Optional[CrossEncoder]:
        """
        Get loaded reranker model.
        
        Returns:
            CrossEncoder model or None if not loaded
        """
        return self._reranker
    
    def is_loaded(self) -> bool:
        """Check if reranker is loaded."""
        return self._reranker is not None
    
    def get_model_name(self) -> Optional[str]:
        """Get loaded model name."""
        return self._model_name
    
    def clear(self) -> None:
        """Clear cached model (for testing or memory management)."""
        with self._lock:
            self._reranker = None
            self._model_name = None


reranker_cache = RerankerCache()


def preload_reranker(model_name: str = "BAAI/bge-reranker-large") -> bool:
    """
    Convenience function to preload reranker at application startup.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        True if preloaded successfully, False otherwise
    """
    try:
        model = reranker_cache.preload(model_name)
        return model is not None
    except Exception as e:
        logger.error(f"Failed to preload reranker: {e}")
        return False
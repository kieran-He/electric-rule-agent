"""
Embedding Cache Singleton for Preloading

Preloads embedding model at application startup to avoid first-request latency.
"""
from __future__ import annotations

import threading
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    SentenceTransformer = None

import logging

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Singleton cache for embedding model preloading.
    
    Thread-safe singleton pattern to ensure model is loaded only once.
    Preloading reduces first-request latency from ~10s to instant.
    """
    
    _instance: Optional[EmbeddingCache] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> EmbeddingCache:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._embedder: Optional[SentenceTransformer] = None
                    cls._instance._model_name: Optional[str] = None
        return cls._instance
    
    def preload(self, model_name: str) -> Optional[SentenceTransformer]:
        """
        Preload embedding model.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Loaded SentenceTransformer model or None if unavailable
        """
        if not EMBEDDING_AVAILABLE:
            logger.warning("sentence-transformers not installed, embedding unavailable")
            return None
        
        if self._embedder is not None:
            if self._model_name != model_name:
                logger.info(f"Reloading embedding: {self._model_name} -> {model_name}")
                self._embedder = None
            else:
                logger.debug(f"Embedding model already loaded: {model_name}")
                return self._embedder
        
        with self._lock:
            if self._embedder is None:
                logger.info(f"Preloading embedding model: {model_name}")
                self._embedder = SentenceTransformer(model_name)
                self._model_name = model_name
                logger.info(f"Embedding model preloaded successfully")
        
        return self._embedder
    
    def get(self) -> Optional[SentenceTransformer]:
        """
        Get loaded embedding model.
        
        Returns:
            SentenceTransformer model or None if not loaded
        """
        return self._embedder
    
    def is_loaded(self) -> bool:
        """Check if embedding is loaded."""
        return self._embedder is not None
    
    def get_model_name(self) -> Optional[str]:
        """Get loaded model name."""
        return self._model_name
    
    def clear(self) -> None:
        """Clear cached model (for testing or memory management)."""
        with self._lock:
            self._embedder = None
            self._model_name = None


embedding_cache = EmbeddingCache()


def preload_embedding(model_name: str) -> bool:
    """
    Convenience function to preload embedding at application startup.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        True if preloaded successfully, False otherwise
    """
    try:
        model = embedding_cache.preload(model_name)
        return model is not None
    except Exception as e:
        logger.error(f"Failed to preload embedding: {e}")
        return False
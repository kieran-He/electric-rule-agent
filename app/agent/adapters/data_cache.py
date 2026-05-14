from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DataCache:
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if time.time() - entry["timestamp"] > self._ttl:
            del self._cache[key]
            logger.debug(f"[DataCache] Key expired: {key}")
            return None
        
        entry["access_count"] = entry.get("access_count", 0) + 1
        return entry["data"]
    
    def set(self, key: str, data: Any) -> None:
        if len(self._cache) >= self._max_size:
            self._evict()
        
        self._cache[key] = {
            "data": data,
            "timestamp": time.time(),
            "access_count": 0,
        }
        logger.debug(f"[DataCache] Key set: {key}")
    
    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        self._cache.clear()
        logger.info("[DataCache] Cache cleared")
    
    def _evict(self) -> None:
        if not self._cache:
            return
        
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: (self._cache[k]["access_count"], self._cache[k]["timestamp"]),
        )
        del self._cache[oldest_key]
        logger.debug(f"[DataCache] Evicted key: {oldest_key}")
    
    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl": self._ttl,
        }
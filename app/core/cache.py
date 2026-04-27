from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    value: any
    timestamp: float
    ttl_seconds: int


@dataclass
class QueryCache:
    _lock: Lock = field(default_factory=Lock)
    _cache: Dict[str, CacheEntry] = field(default_factory=dict)
    _max_size: int = 100
    default_ttl: int = 300
    
    def _hash_query(self, query: str, province_codes: list[str], top_k: int) -> str:
        key = json.dumps({
            "query": query.strip().lower(),
            "provinces": sorted(province_codes),
            "top_k": top_k
        }, ensure_ascii=False)
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def get(self, query: str, province_codes: list[str], top_k: int) -> Optional[any]:
        key = self._hash_query(query, province_codes, top_k)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if time.time() - entry.timestamp > entry.ttl_seconds:
                del self._cache[key]
                return None
            logger.debug(f"Cache hit for query: {query[:50]}")
            return entry.value
    
    def set(self, query: str, province_codes: list[str], top_k: int, value: any, ttl: int = None):
        key = self._hash_query(query, province_codes, top_k)
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
                del self._cache[oldest_key]
            
            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl_seconds=ttl or self.default_ttl
            )
    
    def clear(self):
        with self._lock:
            self._cache.clear()
    
    def stats(self) -> dict:
        with self._lock:
            ages = [time.time() - e.timestamp for e in self._cache.values()]
            return {
                "entries": len(self._cache),
                "oldest_age": min(ages) if ages else 0,
            }


query_cache = QueryCache()
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@dataclass
class MetricPoint:
    timestamp: float
    value: float


@dataclass
class MetricsStore:
    _lock: Lock = field(default_factory=Lock)
    _max_samples: int = 100
    _latency_samples: List[MetricPoint] = field(default_factory=list)
    _retrieval_latency: List[MetricPoint] = field(default_factory=list)
    _llm_latency: List[MetricPoint] = field(default_factory=list)
    _token_counts: List[MetricPoint] = field(default_factory=list)
    _query_counts: Dict[str, int] = field(default_factory=dict)
    _error_counts: Dict[str, int] = field(default_factory=dict)
    _cache_stats: Dict[str, dict] = field(default_factory=dict)
    
    def record_latency(self, latency_ms: float, category: str = "total"):
        with self._lock:
            point = MetricPoint(timestamp=time.time(), value=latency_ms)
            if category == "total":
                self._latency_samples.append(point)
                if len(self._latency_samples) > self._max_samples:
                    self._latency_samples = self._latency_samples[-self._max_samples:]
            elif category == "retrieval":
                self._retrieval_latency.append(point)
                if len(self._retrieval_latency) > self._max_samples:
                    self._retrieval_latency = self._retrieval_latency[-self._max_samples:]
            elif category == "llm":
                self._llm_latency.append(point)
                if len(self._llm_latency) > self._max_samples:
                    self._llm_latency = self._llm_latency[-self._max_samples:]
    
    def record_tokens(self, input_tokens: int, output_tokens: int):
        with self._lock:
            self._token_counts.append(MetricPoint(
                timestamp=time.time(),
                value=input_tokens + output_tokens
            ))
            if len(self._token_counts) > self._max_samples:
                self._token_counts = self._token_counts[-self._max_samples:]
    
    def record_query(self, province_code: str):
        with self._lock:
            self._query_counts[province_code] = self._query_counts.get(province_code, 0) + 1
    
    def record_error(self, error_type: str):
        with self._lock:
            self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
    
    def record_cache_stats(
        self,
        cache_type: str,
        hits: int,
        misses: int,
        size: int
    ):
        """Record cache statistics."""
        with self._lock:
            if cache_type not in self._cache_stats:
                self._cache_stats[cache_type] = {
                    "total_hits": 0,
                    "total_misses": 0,
                    "current_size": 0,
                }
            
            self._cache_stats[cache_type]["total_hits"] = hits
            self._cache_stats[cache_type]["total_misses"] = misses
            self._cache_stats[cache_type]["current_size"] = size
    
    def get_cache_report(self) -> dict:
        """Get cache report."""
        with self._lock:
            report = {}
            for cache_type, stats in self._cache_stats.items():
                total = stats["total_hits"] + stats["total_misses"]
                hit_rate = stats["total_hits"] / total * 100 if total > 0 else 0
                
                report[cache_type] = {
                    "hits": stats["total_hits"],
                    "misses": stats["total_misses"],
                    "hit_rate": f"{hit_rate:.1f}%",
                    "size": stats["current_size"],
                }
            
            return report
    
    def get_summary(self) -> dict:
        with self._lock:
            def avg(points: List[MetricPoint]) -> float:
                if not points:
                    return 0.0
                return sum(p.value for p in points) / len(points)
            
            summary = {
                "latency_avg_ms": avg(self._latency_samples),
                "retrieval_latency_avg_ms": avg(self._retrieval_latency),
                "llm_latency_avg_ms": avg(self._llm_latency),
                "tokens_avg": avg(self._token_counts),
                "query_counts": dict(self._query_counts),
                "error_counts": dict(self._error_counts),
                "samples_count": len(self._latency_samples),
            }
            
            if self._cache_stats:
                cache_report = {}
                for cache_type, stats in self._cache_stats.items():
                    total = stats["total_hits"] + stats["total_misses"]
                    hit_rate = stats["total_hits"] / total * 100 if total > 0 else 0
                    cache_report[cache_type] = {
                        "hits": stats["total_hits"],
                        "misses": stats["total_misses"],
                        "hit_rate": f"{hit_rate:.1f}%",
                        "size": stats["current_size"],
                    }
                summary["cache_stats"] = cache_report
            
            return summary
    
    def clear(self):
        with self._lock:
            self._latency_samples.clear()
            self._retrieval_latency.clear()
            self._llm_latency.clear()
            self._token_counts.clear()
            self._query_counts.clear()
            self._error_counts.clear()
            self._cache_stats.clear()
    
    def save_to_db(
        self,
        db: "Session",
        trace_id: str,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        retrieval_latency_ms: Optional[int] = None,
        llm_latency_ms: Optional[int] = None,
        total_latency_ms: Optional[int] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        province_code: Optional[str] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        success: bool = True,
    ) -> None:
        from app.db.models import MetricsRecord
        
        record = MetricsRecord(
            trace_id=trace_id,
            session_id=session_id,
            request_id=request_id,
            user_id=user_id,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            province_code=province_code,
            error_type=error_type,
            error_message=error_message,
            success=success,
            created_at=datetime.utcnow(),
        )
        db.add(record)
        db.commit()


metrics_store = MetricsStore()
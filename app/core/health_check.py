from __future__ import annotations

import os
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    component: str
    status: str
    message: str = ""
    latency_ms: int = 0


class HealthChecker:
    def check_all(self) -> dict:
        components = [
            self.check_database(),
            self.check_chroma(),
            self.check_llm(),
            self.check_reranker(),
            self.check_bm25(),
        ]
        
        overall = self._overall_status(components)
        
        return {
            "overall": overall,
            "components": [self._to_dict(c) for c in components]
        }
    
    def _to_dict(self, status: HealthStatus) -> dict:
        return {
            "component": status.component,
            "status": status.status,
            "message": status.message,
            "latency_ms": status.latency_ms,
        }
    
    def check_database(self) -> HealthStatus:
        try:
            from sqlalchemy import text
            from app.db.session import engine
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return HealthStatus("database", "ok", "SQLite connected")
        except Exception as e:
            return HealthStatus("database", "error", str(e)[:100])
    
    def check_chroma(self) -> HealthStatus:
        try:
            from app.core.repository import ChromaPolicyRepository
            from app.config import settings
            repo = ChromaPolicyRepository(
                persist_directory=settings.chroma_path,
                embedding_model_name=settings.embedding_model,
            )
            if repo.ready:
                collections = repo._client.list_collections()
                return HealthStatus("chroma", "ok", f"{len(collections)} collections")
            return HealthStatus("chroma", "error", repo.init_error or "not ready")
        except Exception as e:
            return HealthStatus("chroma", "error", str(e)[:100])
    
    def check_llm(self) -> HealthStatus:
        try:
            from app.core.llm_client import LLMClient
            client = LLMClient(
                api_key=os.getenv("LLM_API_KEY", ""),
                endpoint=os.getenv("LLM_ENDPOINT", ""),
                model=os.getenv("LLM_MODEL", ""),
            )
            if client.ready:
                return HealthStatus("llm", "ok", "API key configured")
            return HealthStatus("llm", "degraded", "No API key, using mock")
        except Exception as e:
            return HealthStatus("llm", "error", str(e)[:100])
    
    def check_reranker(self) -> HealthStatus:
        try:
            from app.langchain.reranker_cache import reranker_cache
            if reranker_cache.is_loaded():
                return HealthStatus("reranker", "ok", f"model: {reranker_cache.get_model_name()}")
            return HealthStatus("reranker", "degraded", "not loaded")
        except Exception as e:
            return HealthStatus("reranker", "error", str(e)[:100])
    
    def check_bm25(self) -> HealthStatus:
        try:
            from app.langchain.bm25_indexer import BM25Indexer
            indexer = BM25Indexer()
            if indexer.is_available():
                return HealthStatus("bm25", "ok", f"{len(indexer.documents)} docs indexed")
            return HealthStatus("bm25", "degraded", "index not built")
        except Exception as e:
            return HealthStatus("bm25", "error", str(e)[:100])
    
    def _overall_status(self, components: list[HealthStatus]) -> str:
        statuses = [c.status for c in components]
        if "error" in statuses:
            return "error"
        if "degraded" in statuses:
            return "degraded"
        return "ok"


health_checker = HealthChecker()
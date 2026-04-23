"""
Hybrid QA Orchestrator with BM25 + BGE Rerank + Query Expansion

Extends LangChainQAOrchestrator with hybrid retrieval capabilities.
Uses configurable parameters from settings.
"""
from __future__ import annotations

import uuid
from typing import Any, List
import logging

from sqlalchemy.orm import Session

from app.config import settings as global_settings
from app.repository import ChromaPolicyRepository, PolicyChunk
from app.schemas.answer import CitationItem, QueryAnswer
from app.schemas.query import QueryRequest
from app.langchain.orchestrator import LangChainQAOrchestrator
from app.langchain.bm25_indexer import BM25Indexer
from app.langchain.hybrid_retriever import HybridRetriever, BGEReranker
from app.langchain.query_expander import QueryExpander
from app.langchain.reranker_cache import preload_reranker

logger = logging.getLogger(__name__)


class HybridQAOrchestrator(LangChainQAOrchestrator):
    """
    Hybrid QA Orchestrator with BM25 + BGE Rerank + Query Expansion.
    
    Inherits from LangChainQAOrchestrator and replaces the retrieval method
    with hybrid retrieval (Vector + BM25 + BGE Rerank).
    
    Configurable parameters from settings:
    - reranker_model: BAAI/bge-reranker-large (default)
    - reranker_preload: Preload model at startup (default true)
    - bm25_k1: Term frequency saturation (default 1.5)
    - bm25_b: Document length normalization (default 0.6)
    - query_expansion: Enable query expansion (default false)
    - query_expansion_method: Expansion method (default synonyms)
    """
    
    def __init__(
        self,
        db: Session,
        settings: Any = None,
        use_hybrid: bool = True,
        vector_top_k: int = None,
        bm25_top_k: int = None,
        final_top_k: int = None,
        disable_thinking: bool = True,
    ):
        self.settings = settings or global_settings
        
        vector_top_k = vector_top_k or self.settings.hybrid_vector_top_k
        bm25_top_k = bm25_top_k or self.settings.hybrid_bm25_top_k
        final_top_k = final_top_k or self.settings.hybrid_final_top_k
        
        super().__init__(db=db, settings=settings, disable_thinking=disable_thinking)
        
        self.use_hybrid = use_hybrid
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        
        if use_hybrid:
            self._init_hybrid_retriever()
        else:
            self.hybrid_retriever = None
    
    def _init_hybrid_retriever(self) -> None:
        """Initialize BM25 indexer and hybrid retriever with config params."""
        try:
            bm25_indexer = BM25Indexer(
                k1=self.settings.bm25_k1,
                b=self.settings.bm25_b,
            )
            doc_count = bm25_indexer.build_index()
            
            if doc_count > 0:
                if self.settings.reranker_preload:
                    preload_reranker(self.settings.reranker_model)
                
                reranker = BGEReranker(
                    model_name=self.settings.reranker_model,
                    max_length=self.settings.reranker_max_length,
                )
                
                query_expander = None
                if self.settings.query_expansion:
                    query_expander = QueryExpander(
                        max_expansions=self.settings.query_expansion_max,
                    )
                
                self.hybrid_retriever = HybridRetriever(
                    vector_repo=self.repo,
                    bm25_indexer=bm25_indexer,
                    reranker=reranker,
                    query_expander=query_expander,
                    vector_top_k=self.vector_top_k,
                    bm25_top_k=self.bm25_top_k,
                    final_top_k=self.final_top_k,
                    use_query_expansion=self.settings.query_expansion,
                    query_expansion_method=self.settings.query_expansion_method,
                    query_expansion_max=self.settings.query_expansion_max,
                )
                
                logger.info(f"Hybrid retriever initialized: {self.hybrid_retriever.get_stats()}")
            else:
                logger.warning("BM25 index empty, falling back to vector retrieval")
                self.hybrid_retriever = None
                
        except Exception as e:
            logger.warning(f"Failed to init hybrid retriever: {e}, using vector only")
            self.hybrid_retriever = None
    
    def _retrieve(
        self,
        query: str,
        province_codes: List[str],
        top_k: int,
    ) -> List[PolicyChunk]:
        """
        Hybrid retrieval: Vector + BM25 + BGE Rerank.
        
        Args:
            query: User query
            province_codes: Province codes to search
            top_k: Number of results
            
        Returns:
            List of PolicyChunk
        """
        if self.hybrid_retriever is not None:
            return self.hybrid_retriever.retrieve(query, province_codes)
        else:
            return super()._retrieve(query, province_codes, top_k)
    
    def run(self, req: QueryRequest) -> QueryAnswer:
        """
        Execute QA flow with hybrid retrieval.
        
        Args:
            req: QueryRequest
            
        Returns:
            QueryAnswer
        """
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        
        chunks = self._retrieve(req.query, req.province_codes, req.top_k)
        
        province_code = req.province_codes[0] if req.province_codes else "SN"
        answer = self._generate_answer(req.query, chunks, province_code)
        
        citations = self._build_citations(chunks) if req.need_citation else []
        used_documents = [c.doc_name for c in citations]
        
        return QueryAnswer(
            answer=answer,
            citations=citations,
            intent="clause_qa",
            confidence=0.8 if chunks else 0.3,
            used_documents=used_documents,
            trace_id=trace_id,
            flow=None,
            warnings=[] if chunks else ["未检索到相关文档"],
        )
    
    def get_retrieval_stats(self) -> dict:
        """Get retrieval statistics."""
        stats = {
            "mode": "hybrid" if self.use_hybrid and self.hybrid_retriever else "vector",
            "final_top_k": self.final_top_k,
        }
        
        if self.hybrid_retriever:
            stats.update(self.hybrid_retriever.get_stats())
        
        stats["config"] = {
            "reranker_model": self.settings.reranker_model,
            "reranker_preload": self.settings.reranker_preload,
            "bm25_k1": self.settings.bm25_k1,
            "bm25_b": self.settings.bm25_b,
            "query_expansion": self.settings.query_expansion,
            "query_expansion_method": self.settings.query_expansion_method,
        }
        
        return stats
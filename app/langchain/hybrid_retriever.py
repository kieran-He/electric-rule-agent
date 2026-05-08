"""
Hybrid Retriever with BGE Rerank and Query Expansion

Combines Vector retrieval (ChromaDB) + BM25 retrieval + BGE Rerank for optimal results.
Uses RerankerCache for preloaded model to reduce latency.
Supports query expansion for improved recall.
"""
from __future__ import annotations

import concurrent.futures
from typing import Dict, List, Optional, Tuple
import logging

from app.core.repository import PolicyChunk, ChromaPolicyRepository
from app.langchain.bm25_indexer import BM25Indexer
from app.langchain.reranker_cache import reranker_cache, RERANKER_AVAILABLE
from app.langchain.query_expander import QueryExpander
from app.langchain.query_rewriter import QueryRewriter, RewriteResult
from dataprocess.bm25_builder import ProvinceBM25Indexer

logger = logging.getLogger(__name__)


class BGEReranker:
    """
    BGE Reranker for semantic re-ranking.
    
    Uses RerankerCache singleton for preloaded model.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = max_length
    
    def _get_reranker(self):
        """Get reranker from cache (preloaded or lazy load)."""
        if not RERANKER_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed")
        
        cached = reranker_cache.get()
        
        if cached is not None and reranker_cache.get_model_name() == self.model_name:
            return cached
        
        return reranker_cache.preload(self.model_name, self.max_length)
    
    def rerank(
        self,
        query: str,
        candidates: List[PolicyChunk],
        top_k: int = 12,
        normalize_scores: bool = False,
    ) -> List[PolicyChunk]:
        """
        Re-rank candidates using BGE reranker.
        
        Args:
            query: User query
            candidates: List of candidate chunks
            top_k: Number of results to return
            normalize_scores: Whether to apply batch normalization (default: False)
            
        Returns:
            Re-ranked list of PolicyChunk with raw reranker scores
        """
        if not candidates:
            return []
        
        if not RERANKER_AVAILABLE:
            logger.warning("Reranker not available, returning candidates as-is")
            return candidates[:top_k]
        
        reranker = self._get_reranker()
        
        pairs = [(query, chunk.text) for chunk in candidates]
        scores = reranker.predict(pairs)
        
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        
        top_indices = ranked_indices[:top_k]
        top_scores = [float(scores[i]) for i in top_indices]
        
        if normalize_scores and len(top_scores) > 1:
            min_score = min(top_scores)
            max_score = max(top_scores)
            score_range = max_score - min_score
            
            if score_range > 0.01:
                top_scores = [(s - min_score) / score_range for s in top_scores]
        
        results = []
        for i, idx in enumerate(top_indices):
            chunk = candidates[idx]
            chunk.score = top_scores[i]
            results.append(chunk)
        
        return results
    
    def is_available(self) -> bool:
        """Check if reranker is available."""
        return RERANKER_AVAILABLE
    
    def get_model_name(self) -> str:
        """Get configured model name."""
        return self.model_name


class HybridRetriever:
    """
    Hybrid Retrieval combining Vector + BM25 + BGE Rerank + Query Expansion.
    
    Pipeline:
    1. Query expansion (optional, +10-15% recall)
    2. Vector retrieval (ChromaDB cosine similarity)
    3. BM25 retrieval (keyword matching with configurable k1, b)
    4. Merge and deduplicate
    5. BGE Rerank for final ranking (preloaded model)
    """
    
    def __init__(
        self,
        vector_repo: ChromaPolicyRepository,
        bm25_indexer: BM25Indexer,
        reranker: Optional[BGEReranker] = None,
        query_expander: Optional[QueryExpander] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        llm_wrapper: Optional["MiniMaxLLMWrapper"] = None,
        vector_top_k: int = 8,
        bm25_top_k: int = 8,
        final_top_k: int = 8,
        use_query_expansion: bool = False,
        query_expansion_method: str = "synonyms",
        query_expansion_max: int = 3,
        use_query_rewrite: bool = False,
        query_rewrite_keep_original: bool = True,
        rejection_threshold: Optional[float] = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.6,
        cache_dir: str = "data/cache",
        use_rrf_fusion: bool = True,
        rrf_k: int = 60,
        rrf_stage1_top_k: int = 15,
        rrf_stage2_top_k: int = 20,
    ):
        from app.langchain.llm import MiniMaxLLMWrapper
        self.vector_repo = vector_repo
        self.bm25_indexer = bm25_indexer
        self.reranker = reranker or BGEReranker()
        self.query_expander = query_expander
        self.query_rewriter = query_rewriter
        self.llm_wrapper = llm_wrapper
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.use_query_expansion = use_query_expansion
        self.query_expansion_method = query_expansion_method
        self.query_expansion_max = query_expansion_max
        self.use_query_rewrite = use_query_rewrite
        self.query_rewrite_keep_original = query_rewrite_keep_original
        self.rejection_threshold = rejection_threshold
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.cache_dir = cache_dir
        self.use_rrf_fusion = use_rrf_fusion
        self.rrf_k = rrf_k
        self.rrf_stage1_top_k = rrf_stage1_top_k
        self.rrf_stage2_top_k = rrf_stage2_top_k
        self._bm25_indexers: Dict[str, ProvinceBM25Indexer] = {}
        self._supported_provinces: Optional[List[str]] = None
        self._last_rewrite_result: Optional[RewriteResult] = None
        self._last_expanded_queries: Optional[List[str]] = None
        
        if use_query_expansion and self.query_expander is None:
            self.query_expander = QueryExpander(
                max_expansions=query_expansion_max,
                llm_wrapper=llm_wrapper,
            )
    
    def _get_bm25_indexer(self, province_code: str) -> ProvinceBM25Indexer:
        if province_code not in self._bm25_indexers:
            indexer = ProvinceBM25Indexer(
                province_code=province_code,
                processed_dir=f"data/processed/{province_code}",
                cache_dir=self.cache_dir,
                k1=self.bm25_k1,
                b=self.bm25_b,
            )
            indexer.build_index()
            self._bm25_indexers[province_code] = indexer
            logger.debug(f"Loaded BM25 indexer for province {province_code}")
        return self._bm25_indexers[province_code]
    
    def _get_supported_provinces(self) -> List[str]:
        """
        Get list of supported provinces from ChromaDB collections.
        
        Collections follow pattern: kb_{province_code.lower()}
        e.g., kb_sn (Shaanxi), kb_sd (Shandong)
        
        Returns:
            List of supported province codes (uppercase), e.g., ["SN"]
        """
        if self._supported_provinces is None:
            try:
                collections = self.vector_repo._client.list_collections()
                self._supported_provinces = [
                    c.name.replace("kb_", "").upper()
                    for c in collections
                    if c.name.startswith("kb_") and c.name != "kb_global"
                ]
                logger.debug(f"Supported provinces: {self._supported_provinces}")
            except Exception as e:
                logger.warning(f"Failed to get supported provinces: {e}")
                self._supported_provinces = []
        return self._supported_provinces
    
    def retrieve(
        self,
        query: str,
        province_codes: List[str],
    ) -> Tuple[List[PolicyChunk], List[str]]:
        """
        Hybrid retrieval pipeline with province detection.
        
        Args:
            query: User query
            province_codes: Default province codes (used if no provinces detected)
            
        Returns:
            (Re-ranked list of PolicyChunk, detected province codes)
        """
        rewritten_queries = self._rewrite_query(query)
        
        detected_codes = province_codes
        if self._last_rewrite_result and self._last_rewrite_result.province_codes:
            detected_codes = self._last_rewrite_result.province_codes
            logger.info(f"Detected provinces from query: {detected_codes}")
        
        if not detected_codes:
            logger.info(f"No provinces detected, using default: {province_codes}")
            detected_codes = province_codes
        
        supported = self._get_supported_provinces()
        valid_codes = [c for c in detected_codes if c in supported]
        if not valid_codes:
            valid_codes = province_codes
            logger.warning(f"Detected provinces {detected_codes} not supported, using default: {province_codes}")
        
        rerank_query = rewritten_queries[-1] if len(rewritten_queries) > 1 else query
        
        queries = self._expand_queries(rewritten_queries)
        
        per_query_results: List[List[PolicyChunk]] = []
        for q in queries:
            query_candidates = self._retrieve_provinces_concurrent(q, valid_codes)
            per_query_results.append(query_candidates)
        
        if self.use_rrf_fusion and len(per_query_results) > 1:
            all_candidates = self._rrf_fusion_stage2(
                per_query_results,
                k=self.rrf_k,
                top_k=self.rrf_stage2_top_k,
            )
        elif self.use_rrf_fusion and len(per_query_results) == 1:
            all_candidates = per_query_results[0]
        else:
            all_candidates = []
            for results in per_query_results:
                all_candidates.extend(results)
            all_candidates = self._merge_and_deduplicate(all_candidates, [])
        
        if self.reranker.is_available() and len(all_candidates) > self.final_top_k:
            final_chunks = self.reranker.rerank(rerank_query, all_candidates, top_k=self.final_top_k)
        else:
            actual_final_k = min(self.final_top_k, len(all_candidates))
            final_chunks = all_candidates[:actual_final_k]
        
        logger.debug(f"Rerank using query: '{rerank_query}' (all queries: {queries})")
        
        if self.rejection_threshold is not None and final_chunks:
            if final_chunks[0].score < self.rejection_threshold:
                logger.info(f"Query rejected: top score {final_chunks[0].score:.3f} < threshold {self.rejection_threshold}")
                return [], valid_codes
        
        return final_chunks, valid_codes
    
    def _retrieve_provinces_concurrent(self, query: str, province_codes: List[str]) -> List[PolicyChunk]:
        all_chunks: List[PolicyChunk] = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._retrieve_province, query, code): code
                for code in province_codes
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    province_code = futures[future]
                    logger.warning(f"Failed to retrieve from province {province_code}: {e}")
        
        return all_chunks
    
    def _retrieve_province(self, query: str, province_code: str) -> List[PolicyChunk]:
        vector_chunks = self.vector_repo.retrieve(
            query=query,
            top_k=self.vector_top_k,
            kb_scope="province",
            province_code=province_code,
        )
        
        try:
            bm25_indexer = self._get_bm25_indexer(province_code)
            bm25_results = bm25_indexer.search(query, top_k=self.bm25_top_k)
            bm25_chunks = []
            for chunk_data, score in bm25_results:
                chunk = PolicyChunk(
                    text=chunk_data["text"],
                    score=float(score),
                    metadata=chunk_data["metadata"],
                )
                bm25_chunks.append(chunk)
        except Exception as e:
            logger.warning(f"BM25 search failed for province {province_code}: {e}, using vector only")
            bm25_chunks = []
        
        if self.use_rrf_fusion:
            return self._rrf_fusion_stage1(
                vector_chunks, bm25_chunks,
                k=self.rrf_k,
                top_k=self.rrf_stage1_top_k,
            )
        else:
            return self._merge_and_deduplicate(vector_chunks, bm25_chunks)
    
    def _rewrite_query(self, query: str) -> List[str]:
        """
        Rewrite query if enabled and triggered.
        
        Args:
            query: Original query
            
        Returns:
            List of queries (original + rewritten if triggered)
        """
        self._last_rewrite_result = None
        
        if not self.use_query_rewrite or self.query_rewriter is None:
            return [query]
        
        try:
            result = self.query_rewriter.rewrite(query)
            self._last_rewrite_result = result
            if result.triggered:
                if self.query_rewrite_keep_original:
                    logger.debug(f"Query rewrite: keeping original + rewritten")
                    return [query, result.rewritten_query]
                else:
                    logger.debug(f"Query rewrite: using rewritten only")
                    return [result.rewritten_query]
            else:
                logger.debug(f"Query rewrite not triggered: {result.trigger_reason}")
                return [query]
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}, using original query")
            return [query]
    
    def _expand_queries(self, queries: List[str]) -> List[str]:
        """
        Expand queries if enabled.
        
        Args:
            queries: List of queries (may include rewritten)
            
        Returns:
            List of expanded queries
        """
        if not self.use_query_expansion or self.query_expander is None:
            return queries
        
        all_expanded = []
        for q in queries:
            try:
                expanded = self.query_expander.expand(q, self.query_expansion_method)
                all_expanded.extend(expanded)
            except Exception as e:
                logger.warning(f"Query expansion failed for '{q}': {e}")
                all_expanded.append(q)
        
        seen = set()
        unique = []
        for q in all_expanded:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        
        logger.debug(f"Queries expanded: {len(queries)} -> {len(unique)}")
        self._last_expanded_queries = unique
        return unique
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query if enabled (legacy method, deprecated).
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries
        """
        if not self.use_query_expansion or self.query_expander is None:
            return [query]
        
        try:
            expanded = self.query_expander.expand(query, self.query_expansion_method)
            logger.debug(f"Query expanded: {query} -> {expanded}")
            return expanded
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original query")
            return [query]
    
    def _merge_and_deduplicate(
        self,
        vector_chunks: List[PolicyChunk],
        bm25_chunks: List[PolicyChunk],
    ) -> List[PolicyChunk]:
        """Merge results and deduplicate by text hash."""
        seen_hashes: set[int] = set()
        candidates: List[PolicyChunk] = []
        
        for chunk in vector_chunks + bm25_chunks:
            chunk_hash = hash(chunk.text[:100])
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                candidates.append(chunk)
        
        return candidates
    
    def _rrf_fusion_stage1(
        self,
        vector_chunks: List[PolicyChunk],
        bm25_chunks: List[PolicyChunk],
        k: int = 60,
        top_k: int = 15,
    ) -> List[PolicyChunk]:
        """
        Stage1 RRF: Single query internal Vector + BM25 fusion.
        
        Formula: RRF(d) = 1/(k + rank_vector) + 1/(k + rank_bm25)
        """
        rrf_scores: Dict[int, float] = {}
        chunk_map: Dict[int, PolicyChunk] = {}
        
        for rank, chunk in enumerate(vector_chunks, start=1):
            chunk_hash = hash(chunk.text[:100])
            if chunk_hash not in rrf_scores:
                rrf_scores[chunk_hash] = 0.0
                chunk_map[chunk_hash] = chunk
            rrf_scores[chunk_hash] += 1.0 / (k + rank)
        
        for rank, chunk in enumerate(bm25_chunks, start=1):
            chunk_hash = hash(chunk.text[:100])
            if chunk_hash not in rrf_scores:
                rrf_scores[chunk_hash] = 0.0
                chunk_map[chunk_hash] = chunk
            rrf_scores[chunk_hash] += 1.0 / (k + rank)
        
        sorted_hashes = sorted(rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True)
        
        results = []
        actual_top_k = min(top_k, len(sorted_hashes))
        for h in sorted_hashes[:actual_top_k]:
            chunk = chunk_map[h]
            chunk.score = rrf_scores[h]
            results.append(chunk)
        
        return results
    
    def _rrf_fusion_stage2(
        self,
        per_query_results: List[List[PolicyChunk]],
        k: int = 60,
        top_k: int = 20,
    ) -> List[PolicyChunk]:
        """
        Stage2 RRF: Cross-query aggregation fusion.
        
        Each query's internal RRF results as input.
        Formula: RRF(d) = sum(1/(k + rank_query_i)) for all queries
        """
        rrf_scores: Dict[int, float] = {}
        chunk_map: Dict[int, PolicyChunk] = {}
        
        for query_results in per_query_results:
            for rank, chunk in enumerate(query_results, start=1):
                chunk_hash = hash(chunk.text[:100])
                if chunk_hash not in rrf_scores:
                    rrf_scores[chunk_hash] = 0.0
                    chunk_map[chunk_hash] = chunk
                rrf_scores[chunk_hash] += 1.0 / (k + rank)
        
        sorted_hashes = sorted(rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True)
        
        results = []
        actual_top_k = min(top_k, len(sorted_hashes))
        for h in sorted_hashes[:actual_top_k]:
            chunk = chunk_map[h]
            chunk.score = rrf_scores[h]
            results.append(chunk)
        
        logger.debug(f"RRF Stage2: {len(sorted_hashes)} candidates -> {len(results)} results (top_k={top_k})")
        return results
    
    def get_stats(self) -> dict:
        """Get retrieval statistics."""
        stats = {
            "vector_top_k": self.vector_top_k,
            "bm25_top_k": self.bm25_top_k,
            "final_top_k": self.final_top_k,
            "bm25_available": self.bm25_indexer.is_available(),
            "bm25_params": self.bm25_indexer.get_stats(),
            "reranker_available": self.reranker.is_available(),
            "reranker_model": self.reranker.get_model_name(),
            "query_expansion": self.use_query_expansion,
            "query_expansion_method": self.query_expansion_method,
            "query_rewrite": self.use_query_rewrite,
            "query_rewrite_keep_original": self.query_rewrite_keep_original,
            "rejection_threshold": self.rejection_threshold,
            "llm_available": self.llm_wrapper is not None,
            "use_rrf_fusion": self.use_rrf_fusion,
            "rrf_k": self.rrf_k,
            "rrf_stage1_top_k": self.rrf_stage1_top_k,
            "rrf_stage2_top_k": self.rrf_stage2_top_k,
        }
        
        if self.query_expander:
            stats["query_expander_stats"] = self.query_expander.get_stats()
        
        if self.query_rewriter:
            stats["query_rewriter_stats"] = self.query_rewriter.get_stats()
        
        return stats
    
    def get_last_rewrite_result(self) -> Optional[RewriteResult]:
        """Get the last rewrite result from the most recent retrieve call."""
        return self._last_rewrite_result
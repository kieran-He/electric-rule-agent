"""
Hybrid QA Orchestrator with BM25 + BGE Rerank + Query Expansion

Standalone orchestrator with hybrid retrieval capabilities.
Uses configurable parameters from settings.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, List, Tuple, TYPE_CHECKING
import logging

from sqlalchemy.orm import Session

from app.config import settings as global_settings
from app.core.metrics import metrics_store
from app.core.repository import ChromaPolicyRepository, PolicyChunk
from app.schemas.answer import CitationItem, QueryAnswer
from app.schemas.query import QueryRequest
from app.langchain.bm25_indexer import BM25Indexer
from app.langchain.hybrid_retriever import HybridRetriever, BGEReranker
from app.langchain.query_expander import QueryExpander
from app.langchain.query_rewriter import QueryRewriter
from app.langchain.reranker_cache import preload_reranker
from app.langchain.llm import MiniMaxLLMWrapper
if TYPE_CHECKING:
    from app.services.trace_service import TraceService

logger = logging.getLogger(__name__)


class HybridQAOrchestrator:
    """
    Hybrid QA Orchestrator with BM25 + BGE Rerank + Query Expansion.
    
    Configurable parameters from settings:
    - reranker_model: BAAI/bge-reranker-large (default)
    - reranker_preload: Preload model at startup (default true)
    - bm25_k1: Term frequency saturation (default 1.5)
    - bm25_b: Document length normalization (default 0.6)
    - query_expansion: Enable query expansion (default false)
    - query_expansion_method: Expansion method (default synonyms)
    
    Observability Storage:
    - metrics_record: Aggregated metrics for historical queries and performance analysis
    - trace_record: Detailed per-request traces with query content and retrieved docs
    """
    
    def __init__(
        self,
        db: Session = None,
        settings: Any = None,
        use_hybrid: bool = True,
        vector_top_k: int = None,
        bm25_top_k: int = None,
        final_top_k: int = None,
        disable_thinking: bool = True,
    ):
        self.db = db
        self.settings = settings or global_settings
        
        vector_top_k = vector_top_k or self.settings.hybrid_vector_top_k
        bm25_top_k = bm25_top_k or self.settings.hybrid_bm25_top_k
        final_top_k = final_top_k or self.settings.hybrid_final_top_k
        
        self.repo = ChromaPolicyRepository(
            persist_directory=self.settings.chroma_path,
            embedding_model_name=self.settings.embedding_model,
        )
        
        self.llm_wrapper = MiniMaxLLMWrapper(
            api_key=os.getenv("LLM_API_KEY", ""),
            endpoint=os.getenv("LLM_ENDPOINT", "https://api.minimaxi.com/anthropic"),
            model=os.getenv("LLM_MODEL", "MiniMax-M2.7"),
            disable_thinking=disable_thinking,
        )
        
        self.use_hybrid = use_hybrid
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        
        from app.core.web_search import create_web_search_client
        self.web_search_client = create_web_search_client(self.settings)
        
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
                        llm_wrapper=self.llm_wrapper,
                    )
                
                query_rewriter = None
                if self.settings.query_rewrite_enabled:
                    try:
                        query_rewriter = QueryRewriter(
                            llm_wrapper=self.llm_wrapper,
                            enabled=True,
                            always_rewrite=self.settings.query_rewrite_always,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to init QueryRewriter: {e}")
                
                self.hybrid_retriever = HybridRetriever(
                    vector_repo=self.repo,
                    bm25_indexer=bm25_indexer,
                    reranker=reranker,
                    query_expander=query_expander,
                    query_rewriter=query_rewriter,
                    llm_wrapper=self.llm_wrapper,
                    vector_top_k=self.vector_top_k,
                    bm25_top_k=self.bm25_top_k,
                    final_top_k=self.final_top_k,
                    use_query_expansion=self.settings.query_expansion,
                    query_expansion_method=self.settings.query_expansion_method,
                    query_expansion_max=self.settings.query_expansion_max,
                    use_query_rewrite=self.settings.query_rewrite_enabled,
                    query_rewrite_keep_original=self.settings.query_rewrite_keep_original,
                    bm25_k1=self.settings.bm25_k1,
                    bm25_b=self.settings.bm25_b,
                    cache_dir="data/cache",
                    use_rrf_fusion=self.settings.use_rrf_fusion,
                    rrf_k=self.settings.rrf_k,
                    rrf_stage1_top_k=self.settings.rrf_stage1_top_k,
                    rrf_stage2_top_k=self.settings.rrf_stage2_top_k,
                )
                
                logger.info(f"Hybrid retriever initialized: {self.hybrid_retriever.get_stats()}")
            else:
                logger.warning("BM25 index empty, falling back to vector retrieval")
                self.hybrid_retriever = None
                
        except Exception as e:
            logger.warning(f"Failed to init hybrid retriever: {e}, using vector only")
            self.hybrid_retriever = None
    
    def _retrieve_vector(
        self,
        query: str,
        province_codes: List[str],
        top_k: int,
    ) -> List[PolicyChunk]:
        """Fallback vector-only retrieval."""
        all_chunks: List[PolicyChunk] = []
        for province_code in province_codes:
            chunks = self.repo.retrieve(
                query=query,
                top_k=top_k,
                kb_scope="province",
                province_code=province_code,
            )
            all_chunks.extend(chunks)
        
        seen_hashes: set[int] = set()
        unique_chunks: List[PolicyChunk] = []
        for chunk in all_chunks:
            text_hash = hash(chunk.text[:100])
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks[:top_k]
    
    def _retrieve(
        self,
        query: str,
        province_codes: List[str],
        top_k: int,
    ) -> Tuple[List[PolicyChunk], List[str]]:
        """
        Hybrid retrieval: Vector + BM25 + BGE Rerank with province detection.
        
        Args:
            query: User query
            province_codes: Province codes to search (default/fallback)
            top_k: Number of results
            
        Returns:
            Tuple of (List of PolicyChunk, detected province codes)
        """
        if self.hybrid_retriever is not None:
            return self.hybrid_retriever.retrieve(query, province_codes)
        else:
            fallback_codes = self._detect_provinces_fallback(query, province_codes)
            return self._retrieve_vector(query, province_codes, top_k), fallback_codes
    
    def _detect_provinces_fallback(self, query: str, default_codes: List[str]) -> List[str]:
        """Fallback province detection when hybrid retriever not available."""
        from dataprocess.province_mapping import PROVINCE_ALIASES
        detected = []
        for alias, code in PROVINCE_ALIASES.items():
            if alias in query:
                detected.append(code)
        return detected if detected else default_codes
    
    def _contains_insufficient_evidence(self, answer: str) -> bool:
        """Check if answer indicates insufficient evidence."""
        keywords_str = getattr(self.settings, 'insufficient_evidence_keywords', 
                               "未检索到充分依据,证据不足,未找到相关信息,无法确定,未找到充分证据,知识库中无相关")
        keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
        return any(kw in answer for kw in keywords)
    
    def _generate_answer_with_tokens(
        self,
        query: str,
        chunks: List[PolicyChunk],
        province_code: str,
        history: list[str] = None,
    ) -> tuple[str, int, int]:
        """
        Generate answer using LangChain LLM with token counts.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            province_code: Province code for context
            history: Conversation history list
            
        Returns:
            Tuple of (answer string, input_tokens, output_tokens)
        """
        from app.langchain.retriever_wrapper import format_chunks_for_context
        
        if not chunks:
            return self._web_search_fallback(query), 0, 0
        
        provincial_context = format_chunks_for_context(chunks)
        global_context = "- 无通用证据"
        history_text = "\n".join(history[-6:]) if history else ""
        
        user_content = f"""问题: {query}

省级证据({province_code}):
{provincial_context}

通用证据:
{global_context}

历史对话:
{history_text}

请根据上述证据回答问题。"""
        
        system_prompt = """你是电力政策问答助手。只能根据提供的证据回答，禁止编造。如果证据不足，明确说明"未检索到充分依据"。

回答格式要求：
1. 直接回答用户问题，不要标注来源编号或引用出处
2. 结构清晰，使用标题和列表组织内容
3. 如需引用原文，使用引用格式（> 引用内容）
4. 证据不足时，明确告知用户并建议补充检索
5. 涉及多省份时，分别说明各省份政策"""
        
        try:
            answer, input_tokens, output_tokens = self.llm_wrapper.invoke(user_content, system=system_prompt)
            if not answer:
                return self._build_mock_answer(query, chunks), input_tokens, output_tokens
            
            if self._contains_insufficient_evidence(answer):
                web_search_enabled = getattr(self.settings, 'web_search_on_insufficient_evidence', True)
                if web_search_enabled:
                    logger.info(f"Evidence insufficient detected, triggering web search for: {query}")
                    web_answer = self._web_search_fallback(query)
                    combined_answer = f"{answer}\n\n---\n\n**补充信息（来自网络搜索）：**\n{web_answer}"
                    return combined_answer, input_tokens, output_tokens
            
            return answer, input_tokens, output_tokens
        except Exception as e:
            logger.error(f"LLM invoke failed: {e}")
            return self._build_mock_answer(query, chunks) + f"\n\n[LLM服务暂时不可用: {str(e)[:100]}]", 0, 0
    
    def _web_search_fallback(self, query: str) -> str:
        """
        Fallback to web search when no relevant documents found.
        
        Uses Tavily API for real-time web search (if configured),
        otherwise uses LLM to attempt answering without evidence.
        
        Args:
            query: User query
            
        Returns:
            Answer string with disclaimer about non-knowledge-base content
        """
        logger.info(f"No chunks found, falling back to web search for query: {query}")
        
        if self.web_search_client and self.web_search_client.is_available():
            try:
                results = self.web_search_client.search(query)
                if results:
                    context = self.web_search_client.format_results_for_context(results)
                    
                    system_prompt = """你是电力政策问答助手，根据网络搜索结果回答用户问题。

回答要求：
1. 基于搜索结果回答，不要编造信息
2. 禁止提及来源、证据出处等引用信息
3. 简洁清晰，直接回答问题核心
4. 如搜索结果不足以回答，明确说明"""
                    
                    user_content = f"""问题：{query}

网络搜索结果：
{context}

请根据上述搜索结果回答问题。"""
                    
                    answer, _, _ = self.llm_wrapper.invoke(user_content, system=system_prompt)
                    if answer:
                        return f"⚠️ 此回答来自网络搜索，非知识库内容，仅供参考。\n\n{answer}"
            except Exception as e:
                logger.error(f"Web search with Tavily failed: {e}")
        
        system_prompt = "你是搜索助手，帮助用户从网络获取信息。请尝试回答问题，如无法回答请明确说明。"
        user_content = f"请回答以下问题：{query}\n\n注意：如果无法确定答案，请明确说明。"
        
        try:
            answer, _, _ = self.llm_wrapper.invoke(user_content, system=system_prompt)
            if answer:
                return f"⚠️ 知识库无相关文档，此回答为LLM尝试回答，仅供参考。\n\n{answer}"
            return "未检索到相关文档，也无法通过网络搜索获取答案。请尝试更换关键词或联系管理员确认文档库是否完整。"
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return "未检索到相关文档，网络搜索服务暂时不可用。请尝试更换关键词或联系管理员确认文档库是否完整。"
    
    def _build_mock_answer(self, query: str, chunks: List[PolicyChunk]) -> str:
        """Build mock answer when LLM unavailable."""
        if not chunks:
            return "未检索到相关文档。"
        
        lines = [f"关于您的问题「{query}」，根据检索到的文档："]
        for i, chunk in enumerate(chunks[:3], 1):
            source = chunk.metadata.get("source_name", "未知文档")
            title_path = chunk.metadata.get("title_path", "")
            article_no = chunk.metadata.get("article_no", "")
            snippet = chunk.text[:200]
            lines.append(f"\n{i}. {source}")
            if title_path:
                lines.append(f"   位置: {title_path}")
            if article_no:
                lines.append(f"   条款: {article_no}")
            lines.append(f"   内容摘要: {snippet}...")
        
        return "\n".join(lines)
    
    def _build_citations(self, chunks: List[PolicyChunk]) -> List[CitationItem]:
        """Build citation items from chunks."""
        citations: List[CitationItem] = []
        for chunk in chunks[:8]:
            citation = CitationItem(
                doc_name=chunk.metadata.get("doc_name") or chunk.metadata.get("source_name", ""),
                status=chunk.metadata.get("policy_level", "formal"),
                title_path=chunk.metadata.get("title_path", ""),
                article_no=chunk.metadata.get("article_no"),
                excerpt=chunk.text[:260],
                page_start=int(chunk.metadata.get("page_start", 0)) if chunk.metadata.get("page_start") else None,
                page_end=int(chunk.metadata.get("page_end", 0)) if chunk.metadata.get("page_end") else None,
            )
            citations.append(citation)
        return citations
    
    def run(self, req: QueryRequest, history: list[str] = None, trace_service: "TraceService" = None, db: Session = None) -> QueryAnswer:
        """
        Execute QA flow with hybrid retrieval.
        
        Args:
            req: QueryRequest
            history: Conversation history list
            trace_service: TraceService for recording
            
        Returns:
            QueryAnswer with detected provinces info
        """
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        
        retrieval_start = time.time()
        chunks, detected_codes = self._retrieve(req.query, req.province_codes, req.top_k)
        retrieval_latency = int((time.time() - retrieval_start) * 1000)
        metrics_store.record_latency(retrieval_latency, "retrieval")
        
        rewrite_result = None
        if self.hybrid_retriever:
            rewrite_result = self.hybrid_retriever.get_last_rewrite_result()
        
        province_code = detected_codes[0] if detected_codes else "SN"
        
        llm_start = time.time()
        answer, input_tokens, output_tokens = self._generate_answer_with_tokens(
            req.query, chunks, province_code, history or []
        )
        llm_latency = int((time.time() - llm_start) * 1000)
        metrics_store.record_latency(llm_latency, "llm")
        metrics_store.record_tokens(input_tokens, output_tokens)
        
        citations = self._build_citations(chunks) if req.need_citation else []
        used_documents = [c.doc_name for c in citations]
        
        total_latency = int((time.time() - start_time) * 1000)
        metrics_store.record_latency(total_latency, "total")
        metrics_store.record_query(province_code)
        
        db_session = db or self.db
        try:
            metrics_store.save_to_db(
                db=db_session,
                trace_id=trace_id,
                session_id=req.session_id,
                request_id=getattr(req, 'request_id', None),
                user_id=getattr(req, 'user_id', None),
                retrieval_latency_ms=retrieval_latency,
                llm_latency_ms=llm_latency,
                total_latency_ms=total_latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                province_code=province_code,
                success=True,
            )
        except Exception as e:
            logger.warning(f"Failed to save metrics to db: {e}")
        
        if trace_service:
            trace_service.save_trace(
                trace_id=trace_id,
                session_id=req.session_id,
                raw_query=req.query,
                rewritten_query=rewrite_result.rewritten_query if rewrite_result and rewrite_result.triggered else None,
                intent="clause_qa",
                retrieved_doc_ids=[int(c.metadata.get("doc_id", 0)) for c in chunks if c.metadata.get("doc_id")],
                rerank_scores=[c.score for c in chunks],
                latency_ms=total_latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                retrieval_latency_ms=retrieval_latency,
                llm_latency_ms=llm_latency,
                retrieved_doc_texts=[c.text[:500] for c in chunks],
                answer_text=answer,
            )
        
        detected_provinces_str = ", ".join(detected_codes) if detected_codes else ""
        
        return QueryAnswer(
            answer=answer,
            citations=citations,
            intent="clause_qa",
            confidence=0.8 if chunks else 0.3,
            used_documents=used_documents,
            trace_id=trace_id,
            flow=None,
            warnings=[] if chunks else ["未检索到相关文档"],
            detected_provinces=detected_provinces_str,
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
            "query_rewrite": self.settings.query_rewrite_enabled,
            "query_rewrite_min_length": self.settings.query_rewrite_min_length,
        }
        
        return stats
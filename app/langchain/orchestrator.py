"""
LangChain-based QA Orchestrator

Replaces QAOrchestrator with LCEL chain-based implementation,
with Agent expansion capability reserved.
"""
import os
import uuid
from typing import Any, List, Optional

from sqlalchemy.orm import Session

from app.config import settings as global_settings
from app.repository import ChromaPolicyRepository, PolicyChunk
from app.schemas.answer import CitationItem, QueryAnswer
from app.schemas.query import QueryRequest
from app.langchain.llm import create_minimax_llm, MiniMaxLLMWrapper
from app.langchain.retriever_wrapper import ChromaRepositoryRetriever, format_chunks_for_context


class LangChainQAOrchestrator:
    """
    LangChain-based QA Orchestrator with Agent expansion support.

    Supports switching between Chain mode and Agent mode (future).
    """

    def __init__(
        self,
        db: Session,
        settings: Any = None,
        use_agent: bool = False,
        disable_thinking: bool = True,
    ):
        self.db = db
        self._settings = settings or global_settings
        self.use_agent = use_agent
        self.disable_thinking = disable_thinking

        # Initialize repository
        self.repo = ChromaPolicyRepository(
            persist_directory=self._settings.chroma_path,
            embedding_model_name=self._settings.embedding_model,
        )

        # Initialize LangChain LLM
        self.llm_wrapper = MiniMaxLLMWrapper(
            api_key=os.getenv("LLM_API_KEY", ""),
            endpoint=os.getenv("LLM_ENDPOINT", "https://api.minimaxi.com/anthropic"),
            model=os.getenv("LLM_MODEL", "MiniMax-M2.7"),
            disable_thinking=disable_thinking,
        )

        self.langchain_llm = create_minimax_llm(
            api_key=os.getenv("LLM_API_KEY", ""),
            endpoint=os.getenv("LLM_ENDPOINT", "https://api.minimaxi.com/anthropic"),
            model=os.getenv("LLM_MODEL", "MiniMax-M2.7"),
        )

        if use_agent:
            self._runner = self._build_agent()
        else:
            self._runner = self._build_chain_runner()

    def _build_chain_runner(self):
        """Build Chain mode runner (current implementation)."""
        return None  # Chain runner built per request

    def _build_agent(self):
        """
        Build Agent mode runner (future implementation).

        Reserved for future expansion with tools support.
        """
        # TODO: Implement Agent mode with tools
        # from langchain.agents import create_structured_chat_agent
        # tools = [RetrieverTool(), CalculatorTool(), ...]
        # return create_structured_chat_agent(self.langchain_llm, tools, ...)
        return None

    def run(self, req: QueryRequest) -> QueryAnswer:
        """
        Execute QA flow using LangChain components.

        Args:
            req: QueryRequest with query and parameters

        Returns:
            QueryAnswer with answer and citations
        """
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"

        # Step 1: Retrieve chunks
        chunks = self._retrieve(req.query, req.province_codes, req.top_k)

        # Step 2: Generate answer using LangChain LLM
        province_code = req.province_codes[0] if req.province_codes else "SN"
        answer = self._generate_answer(req.query, chunks, province_code)

        # Step 3: Build citations
        citations = self._build_citations(chunks) if req.need_citation else []

        # Step 4: Build response
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

    def _retrieve(
        self,
        query: str,
        province_codes: List[str],
        top_k: int,
    ) -> List[PolicyChunk]:
        """
        Retrieve chunks from ChromaPolicyRepository.

        Args:
            query: User query
            province_codes: List of province codes to search
            top_k: Number of chunks per province

        Returns:
            Deduplicated list of PolicyChunk
        """
        all_chunks: List[PolicyChunk] = []
        for province_code in province_codes:
            chunks = self.repo.retrieve(
                query=query,
                top_k=top_k,
                kb_scope="province",
                province_code=province_code,
            )
            all_chunks.extend(chunks)

        # Deduplicate using hash (faster than string comparison)
        seen_hashes: set[int] = set()
        unique_chunks: List[PolicyChunk] = []
        for chunk in all_chunks:
            text_hash = hash(chunk.text[:100])
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_chunks.append(chunk)

        return unique_chunks[:top_k]

    def _generate_answer(
        self,
        query: str,
        chunks: List[PolicyChunk],
        province_code: str,
    ) -> str:
        """
        Generate answer using LangChain LLM.

        Args:
            query: User query
            chunks: Retrieved chunks
            province_code: Province code for context

        Returns:
            Generated answer string
        """
        if not chunks:
            return "未检索到相关文档，无法回答该问题。请尝试更换关键词或联系管理员确认文档库是否完整。"

        # Build context
        provincial_context = format_chunks_for_context(chunks)
        global_context = "- 无通用证据"
        history = ""

        # Build prompt
        user_content = f"""问题: {query}

省级证据({province_code}):
{provincial_context}

通用证据:
{global_context}

历史对话:
{history}

请根据上述证据回答问题。"""

        system_prompt = """你是电力政策问答助手。只能根据提供的证据回答，禁止编造。如果证据不足，明确说明"未检索到充分依据"。

回答要求：
1. 基于证据内容回答，不要添加证据中没有的信息
2. 引用证据时标注来源文档名称
3. 如果问题涉及多个省份，分别说明各省份的政策
4. 如果证据不足，明确告知用户并建议补充检索"""

        try:
            answer = self.llm_wrapper.invoke(user_content, system=system_prompt)
            if not answer:
                return self._build_mock_answer(query, chunks)
            return answer
        except Exception as e:
            return self._build_mock_answer(query, chunks) + f"\n\n[LLM服务暂时不可用: {str(e)[:100]}]"

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
        for chunk in chunks[:5]:
            citation = CitationItem(
                doc_name=chunk.metadata.get("source_name", ""),
                status=chunk.metadata.get("policy_level", "formal"),
                title_path=chunk.metadata.get("title_path", ""),
                article_no=chunk.metadata.get("article_no"),
                excerpt=chunk.text[:260],
                page_start=int(chunk.metadata.get("page_start", 0)) if chunk.metadata.get("page_start") else None,
                page_end=int(chunk.metadata.get("page_end", 0)) if chunk.metadata.get("page_end") else None,
            )
            citations.append(citation)
        return citations
    
    def get_retrieval_stats(self) -> dict:
        """Get retrieval statistics."""
        return {
            "mode": "vector",
            "embedder": self.repo.embedder_name,
            "repo_ready": self.repo.ready,
        }

    def run_compare(self, req: QueryRequest) -> QueryAnswer:
        """
        Execute cross-province comparison.

        Args:
            req: QueryRequest with multiple province_codes

        Returns:
            QueryAnswer with comparison results
        """
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"

        # Retrieve per province
        result_by_province = {}
        for province_code in req.province_codes:
            chunks = self.repo.retrieve(
                query=req.query,
                top_k=req.top_k,
                kb_scope="province",
                province_code=province_code,
            )
            result_by_province[province_code] = chunks

        # Build cross-province context
        lines = [f"问题: {req.query}", "跨省检索证据:"]
        for province, chunks in result_by_province.items():
            lines.append(f"\n=== {province} ===")
            lines.append(format_chunks_for_context(chunks))

        user_content = "\n".join(lines)
        system_prompt = """你是电力政策跨省对比分析助手。请基于给定的跨省证据输出结论与差异点。

分析要求：
1. 分别总结各省份的相关政策要点
2. 指出各省份政策的共同点和差异点
3. 如果某省份没有相关证据，明确说明"该省份未检索到相关依据"
4. 不要编造证据中没有的内容"""

        try:
            answer = self.llm_wrapper.invoke(user_content, system=system_prompt)
        except Exception as e:
            answer = f"跨省对比分析暂时不可用: {str(e)[:100]}"

        # Build citations from all provinces
        all_chunks = []
        for chunks in result_by_province.values():
            all_chunks.extend(chunks)

        citations = self._build_citations(all_chunks) if req.need_citation else []
        used_documents = [c.doc_name for c in citations]

        return QueryAnswer(
            answer=answer,
            citations=citations,
            intent="cross_province_compare",
            confidence=0.7,
            used_documents=used_documents,
            trace_id=trace_id,
            flow=None,
            warnings=[],
        )
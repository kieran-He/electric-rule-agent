from __future__ import annotations

import os
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings as global_settings
from app.generator import LLMClient, LLMGenerationError
from app.repository import ChromaPolicyRepository, PolicyChunk
from app.schemas.answer import CitationItem, QueryAnswer
from app.schemas.query import QueryRequest


class QAOrchestrator:
    def __init__(self, db: Session, settings: Any = None):
        self.db = db
        self._settings = settings or global_settings
        self.repo = ChromaPolicyRepository(
            persist_directory=self._settings.chroma_path,
            embedding_model_name=self._settings.embedding_model,
        )
        self.llm_client = LLMClient(
            api_key=os.getenv("LLM_API_KEY", ""),
            endpoint=os.getenv("LLM_ENDPOINT", "https://api.minimaxi.com/anthropic"),
            model=os.getenv("LLM_MODEL", "minimax-2.7"),
            timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
            provider=os.getenv("LLM_PROVIDER", "anthropic"),
        )

    def run(self, req: QueryRequest) -> QueryAnswer:
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        
        chunks = self._retrieve(req.query, req.province_codes, req.top_k)
        
        answer = self._generate_answer(req.query, chunks, req.province_codes[0] if req.province_codes else "SN")
        
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

    def _retrieve(self, query: str, province_codes: list[str], top_k: int) -> list[PolicyChunk]:
        all_chunks: list[PolicyChunk] = []
        for province_code in province_codes:
            chunks = self.repo.retrieve(
                query=query,
                top_k=top_k,
                kb_scope="province",
                province_code=province_code,
            )
            all_chunks.extend(chunks)
        
        seen_hashes: set[int] = set()
        unique_chunks: list[PolicyChunk] = []
        for chunk in all_chunks:
            text_hash = hash(chunk.text[:100])
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks[:top_k]

    def _generate_answer(self, query: str, chunks: list[PolicyChunk], province_code: str) -> str:
        if not chunks:
            return "未检索到相关文档，无法回答该问题。请尝试更换关键词或联系管理员确认文档库是否完整。"
        
        if not self.llm_client.ready:
            return self._build_mock_answer(query, chunks)
        
        try:
            return self.llm_client.generate_answer(
                query=query,
                provincial_chunks=chunks,
                global_chunks=[],
                history=[],
                province_code=province_code,
            )
        except LLMGenerationError as e:
            return self._build_mock_answer(query, chunks) + f"\n\n[LLM服务暂时不可用: {str(e)}]"

    def _build_mock_answer(self, query: str, chunks: list[PolicyChunk]) -> str:
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

    def _build_citations(self, chunks: list[PolicyChunk]) -> list[CitationItem]:
        citations: list[CitationItem] = []
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
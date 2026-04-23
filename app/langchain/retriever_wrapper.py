"""
Retriever Wrapper for ChromaPolicyRepository

Wraps existing ChromaPolicyRepository as LangChain-compatible Retriever.
"""
from typing import List, Optional

from app.repository import ChromaPolicyRepository, PolicyChunk


class ChromaRepositoryRetriever:
    """
    Wrapper that converts ChromaPolicyRepository to LangChain Retriever interface.

    This allows using the existing optimized repository with LangChain chains.
    """

    def __init__(
        self,
        repo: ChromaPolicyRepository,
        province_codes: Optional[List[str]] = None,
        top_k: int = 5,
        kb_scope: str = "province",
    ):
        self.repo = repo
        self.province_codes = province_codes or ["SN"]
        self.top_k = top_k
        self.kb_scope = kb_scope

    def invoke(self, query: str) -> List[PolicyChunk]:
        """
        Retrieve chunks for a query.

        Args:
            query: User query string

        Returns:
            List of PolicyChunk objects
        """
        all_chunks: List[PolicyChunk] = []

        for province_code in self.province_codes:
            chunks = self.repo.retrieve(
                query=query,
                top_k=self.top_k,
                kb_scope=self.kb_scope,
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

        return unique_chunks[:self.top_k]

    def __call__(self, query: str) -> List[PolicyChunk]:
        """Alias for invoke method."""
        return self.invoke(query)


def format_chunks_for_context(chunks: List[PolicyChunk]) -> str:
    """
    Format PolicyChunk list into context string for LLM.

    Args:
        chunks: List of PolicyChunk objects

    Returns:
        Formatted context string
    """
    if not chunks:
        return "- 无相关证据"

    lines = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source_name", "未知文档")
        title_path = chunk.metadata.get("title_path", "")
        article_no = chunk.metadata.get("article_no", "")
        policy_level = chunk.metadata.get("policy_level", "formal")
        snippet = chunk.text[:260]

        line = f"{i}. [{source}] {snippet}"
        if title_path:
            line += f"\n   位置: {title_path}"
        if article_no:
            line += f"\n   条款: {article_no}"
        if policy_level == "draft":
            line += "\n   [注意: 此为征求意见稿/草案]"
        lines.append(line)

    return "\n".join(lines)
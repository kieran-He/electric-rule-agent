"""
Retriever Wrapper for ChromaPolicyRepository

Wraps existing ChromaPolicyRepository as LangChain-compatible Retriever.
"""
from typing import List, Optional, Tuple, Dict

from app.core.repository import ChromaPolicyRepository, PolicyChunk
from app.core.context_compressor import ContextCompressor


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
        return "- 无相关内容"

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


def format_chunks_for_context_with_compression(
    chunks: List[PolicyChunk],
    compress: bool = True,
    max_chars: int = 3000,
    show_chunks: bool = True,
) -> Tuple[str, Dict]:
    """
    格式化检索结果为上下文，支持压缩

    Args:
        chunks: 检索结果
        compress: 是否启用压缩
        max_chars: 最大上下文字符数
        show_chunks: 是否添加chunk编号标记（用于正文引用）

    Returns:
        (context_string, compression_stats)
    """
    if not chunks:
        return "- 无相关内容", {"original": 0, "compressed": 0, "reduction": 0.0}

    if compress:
        compressor = ContextCompressor(max_chars=max_chars)
        compressed = compressor.compress(chunks)
        chunks = compressed.compressed_chunks
        stats = {
            "original": compressed.original_count,
            "compressed": compressed.compressed_count,
            "reduction": compressed.reduction_ratio,
        }
    else:
        stats = {"original": len(chunks), "compressed": len(chunks), "reduction": 0.0}

    lines = []
    for i, chunk in enumerate(chunks, 1):
        policy_level = chunk.metadata.get("policy_level", "formal")
        level_mark = " [草案]" if policy_level == "draft" else ""
        doc_name = chunk.metadata.get("doc_name") or chunk.metadata.get("source_name", "未知文档")
        
        if show_chunks:
            # 添加chunk编号标记，便于LLM引用
            short_name = doc_name[:25] if len(doc_name) > 25 else doc_name
            line = f"[chunk-{i}] 《{short_name}》\n{chunk.text}{level_mark}"
        else:
            line = f"{i}. {chunk.text}{level_mark}"
        lines.append(line)

    return "\n\n".join(lines), stats
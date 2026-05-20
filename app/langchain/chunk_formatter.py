"""
Chunk formatter module for adding references to answer.

Formats answer with chunk references and original text at the end.
"""
from __future__ import annotations

from typing import List

from app.core.repository import PolicyChunk


def format_answer_with_chunk_refs(
    answer: str,
    chunks: List[PolicyChunk],
    max_chunks: int = 3,
) -> str:
    """
    Add simplified chunk references to answer end.
    
    Output format:
    ---
    answer正文内容（LLM已在句末插入引用标记 [《文档名》](#chunk-N)）
    
    ---
    
    ## 参考材料
    
    1. 《文档名》
    
    chunk全文...
    
    2. 《文档名》
    
    chunk全文...
    """
    if not chunks:
        return answer
    
    chunks_to_show = chunks[:max_chunks]
    
    ref_parts = []
    for i, chunk in enumerate(chunks_to_show, 1):
        doc_name = chunk.metadata.get("doc_name") or chunk.metadata.get("source_name", "未知文档")
        chunk_text = chunk.text
        ref_parts.append(f"{i}. 《{doc_name}》\n\n{chunk_text}")
    
    ref_section = "## 参考材料\n\n" + "\n\n".join(ref_parts)
    
    return f"{answer}\n\n---\n\n{ref_section}"
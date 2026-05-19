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
    Add chunk references and original text to answer.
    
    Output format:
    ---
    answer正文内容...
    
    ---
    
    **相关参考**：[查看文档1](#chunk-1) | [查看文档2](#chunk-2)
    
    ---
    
    ## 参考材料原文
    
    ### chunk-1
    **来源**：文档名
    **发布单位**：发改委
    **发布日期**：2025-12-07
    
    > chunk原文内容...
    
    ### chunk-2
    ...
    """
    if not chunks:
        return answer
    
    chunks_to_show = chunks[:max_chunks]
    
    ref_links = []
    for i, chunk in enumerate(chunks_to_show, 1):
        doc_name = chunk.metadata.get("doc_name") or chunk.metadata.get("source_name", "未知文档")
        short_name = doc_name[:30] if len(doc_name) > 30 else doc_name
        ref_links.append(f"[查看{short_name}](#chunk-{i})")
    
    ref_section = "**相关参考**：" + " | ".join(ref_links)
    
    original_sections = []
    for i, chunk in enumerate(chunks_to_show, 1):
        doc_name = chunk.metadata.get("doc_name") or chunk.metadata.get("source_name", "未知文档")
        issuer = chunk.metadata.get("issuer")
        issue_date = chunk.metadata.get("issue_date")
        effective_date = chunk.metadata.get("effective_date")
        
        section_lines = [f"### chunk-{i}"]
        section_lines.append(f"**来源**：{doc_name}")
        
        if issuer:
            section_lines.append(f"**发布单位**：{issuer}")
        
        if issue_date and issue_date != "None":
            section_lines.append(f"**发布日期**：{issue_date}")
        
        if effective_date and effective_date != "None":
            section_lines.append(f"**生效日期**：{effective_date}")
        
        section_lines.append("")
        text_display = chunk.text[:800]
        if len(chunk.text) > 800:
            text_display = text_display + "..."
        section_lines.append(f"> {text_display}")
        
        original_sections.append("\n".join(section_lines))
    
    result = f"{answer}\n\n---\n\n{ref_section}\n\n---\n\n## 参考材料原文\n\n" + "\n\n".join(original_sections)
    
    return result
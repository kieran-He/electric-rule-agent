"""
上下文压缩器：优化检索内容，减少冗余
"""
from typing import List
from dataclasses import dataclass
import logging

from app.core.repository import PolicyChunk

logger = logging.getLogger(__name__)


@dataclass
class CompressedContext:
    compressed_chunks: List[PolicyChunk]
    original_count: int
    compressed_count: int
    reduction_ratio: float


class ContextCompressor:
    """
    检索结果压缩器
    
    功能：
    1. 去重：合并高度相似的段落
    2. 截断：按相关性动态截取长度
    3. 排序：按相关性分数排序
    4. 限制：总长度不超过 max_chars
    """
    
    SIMILARITY_THRESHOLD = 0.85
    
    def __init__(
        self,
        max_chars: int = 3000,
        similarity_threshold: float = 0.85,
        min_chunks: int = 6,  # 最少保留的 chunks 数量
    ):
        self.max_chars = max_chars
        self.similarity_threshold = similarity_threshold
        self.min_chunks = min_chunks
    
    def compress(self, chunks: List[PolicyChunk]) -> CompressedContext:
        """
        压缩检索结果
        
        Args:
            chunks: 原始检索结果
            
        Returns:
            CompressedContext 包含压缩后的 chunks 和统计信息
        """
        if not chunks:
            return CompressedContext([], 0, 0, 0.0)
        
        original_count = len(chunks)
        
        # 不再按 score 排序，保持原始检索顺序
        # 这样 context 中的 chunk-N 编号与 chunks 数组索引一致
        # 避免 LLM 引用时文档名与编号不匹配
        # sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        
        deduped_chunks = self._deduplicate(chunks)
        
        truncated_chunks = self._adaptive_truncate(deduped_chunks)
        
        final_chunks = self._limit_total_length(truncated_chunks)
        
        compressed_count = len(final_chunks)
        reduction_ratio = (original_count - compressed_count) / original_count if original_count > 0 else 0.0
        
        logger.info(
            f"Context compressed: {original_count} -> {compressed_count} "
            f"(reduced {reduction_ratio:.1%})"
        )
        
        return CompressedContext(
            compressed_chunks=final_chunks,
            original_count=original_count,
            compressed_count=compressed_count,
            reduction_ratio=reduction_ratio,
        )
    
    def _deduplicate(self, chunks: List[PolicyChunk]) -> List[PolicyChunk]:
        """
        去除高度相似的段落
        
        使用文本前100字符的 Jaccard 相似度
        但同一个文档的多个段落不去重（可能是不同章节）
        """
        if len(chunks) <= 1:
            return chunks
        
        unique_chunks = []
        
        for chunk in chunks:
            is_duplicate = False
            chunk_text_prefix = chunk.text[:100]
            chunk_doc = chunk.metadata.get("doc_name") or chunk.metadata.get("source_name", "")
            
            for existing in unique_chunks:
                existing_prefix = existing.text[:100]
                existing_doc = existing.metadata.get("doc_name") or existing.metadata.get("source_name", "")
                
                # 不同文档不去重（即使文本开头相似）
                if chunk_doc != existing_doc:
                    continue
                
                similarity = self._jaccard_similarity(chunk_text_prefix, existing_prefix)
                
                if similarity >= self.similarity_threshold:
                    if chunk.score > existing.score:
                        unique_chunks.remove(existing)
                        unique_chunks.append(chunk)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        计算 Jaccard 相似度（字符级别）
        """
        if not text1 or not text2:
            return 0.0
        
        set1 = set(text1)
        set2 = set(text2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _adaptive_truncate(self, chunks: List[PolicyChunk]) -> List[PolicyChunk]:
        """
        自适应截取：高相关性保留更多内容
        
        规则：
        - score >= 0.8: 保留 300 字符
        - score >= 0.6: 保留 200 字符
        - score < 0.6:  保留 100 字符
        """
        truncated = []
        
        for chunk in chunks:
            if chunk.score >= 0.8:
                max_len = 300
            elif chunk.score >= 0.6:
                max_len = 200
            else:
                max_len = 100
            
            truncated_text = chunk.text[:max_len]
            new_chunk = PolicyChunk(
                text=truncated_text,
                score=chunk.score,
                metadata=chunk.metadata,
            )
            truncated.append(new_chunk)
        
        return truncated
    
    def _limit_total_length(self, chunks: List[PolicyChunk]) -> List[PolicyChunk]:
        """
        限制总上下文长度，但确保至少保留 min_chunks 个 chunks
        """
        total_chars = 0
        final_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_len = len(chunk.text)
            
            # 确保至少保留 min_chunks 个 chunks
            if i < self.min_chunks:
                # 对于前 min_chunks 个 chunks，如果超长则截断但仍保留
                if total_chars + chunk_len <= self.max_chars:
                    final_chunks.append(chunk)
                    total_chars += chunk_len
                else:
                    remaining = self.max_chars - total_chars
                    if remaining > 50:
                        truncated_chunk = PolicyChunk(
                            text=chunk.text[:remaining],
                            score=chunk.score,
                            metadata=chunk.metadata,
                        )
                        final_chunks.append(truncated_chunk)
                        total_chars += remaining
                    # 即使空间不足，也要确保前 min_chunks 个 chunks 至少保留一部分
                    elif remaining <= 50 and i < self.min_chunks:
                        # 强制保留 50 字符
                        truncated_chunk = PolicyChunk(
                            text=chunk.text[:50],
                            score=chunk.score,
                            metadata=chunk.metadata,
                        )
                        final_chunks.append(truncated_chunk)
                        total_chars += 50
                continue
            
            # 对于后续 chunks，正常处理
            if total_chars + chunk_len <= self.max_chars:
                final_chunks.append(chunk)
                total_chars += chunk_len
            else:
                remaining = self.max_chars - total_chars
                if remaining > 50:
                    truncated_chunk = PolicyChunk(
                        text=chunk.text[:remaining],
                        score=chunk.score,
                        metadata=chunk.metadata,
                    )
                    final_chunks.append(truncated_chunk)
                break
        
        return final_chunks
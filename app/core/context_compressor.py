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
    ):
        self.max_chars = max_chars
        self.similarity_threshold = similarity_threshold
    
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
        
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        
        deduped_chunks = self._deduplicate(sorted_chunks)
        
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
        """
        if len(chunks) <= 1:
            return chunks
        
        unique_chunks = []
        
        for chunk in chunks:
            is_duplicate = False
            chunk_text_prefix = chunk.text[:100]
            
            for existing in unique_chunks:
                existing_prefix = existing.text[:100]
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
        限制总上下文长度
        """
        total_chars = 0
        final_chunks = []
        
        for chunk in chunks:
            chunk_len = len(chunk.text)
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
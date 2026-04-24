"""
LLM-powered Query Rewriter for Enhanced Retrieval

Rewrites ambiguous/colloquial queries into more precise forms for better retrieval.
Uses intelligent triggering to avoid unnecessary LLM calls.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

from app.langchain.llm import MiniMaxLLMWrapper

logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    rewritten_query: str
    confidence: float
    triggered: bool
    trigger_reason: str


class QueryRewriter:
    """
    LLM-powered query rewriter for improved retrieval.
    
    Intelligent triggering: only rewrites ambiguous/colloquial queries
    Parallel retrieval: original and rewritten queries used together
    """
    
    DOMAIN_KEYWORDS = [
        "电力", "交易", "结算", "现货", "中长期", "零售", 
        "辅助", "储能", "调频", "市场", "规则", "政策",
        "电价", "负荷", "发电", "用电", "电网"
    ]
    COLLOQUIAL_PATTERNS = [
        "怎么", "如何", "那个", "这个", "什么", "哪些", 
        "有没有", "能不能", "可以吗", "为什么", "怎样"
    ]
    
    SYSTEM_PROMPT = """你是电力政策领域专家，擅长将用户口语化查询转换为精确的检索查询。

请将用户查询改写为更适合检索的形式，遵循以下规则：
1. 补充领域关键词（如"交易规则" -> "电力市场交易实施细则"）
2. 去除口语化表达（如"怎么" -> 具体操作流程）
3. 添加必要限定词（如省份、政策类型）
4. 保持查询简洁，不要过度扩展

输出JSON格式：
{"rewritten": "改写后的查询", "keywords_added": ["补充的关键词"]}"""

    def __init__(
        self,
        llm_wrapper: Optional[MiniMaxLLMWrapper] = None,
        enabled: bool = True,
        min_length: int = 10,
    ):
        self.llm = llm_wrapper
        self.enabled = enabled
        self.min_length = min_length
    
    def should_rewrite(self, query: str) -> Tuple[bool, str]:
        """
        Determine if query should be rewritten.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (should_rewrite, reason)
        """
        if not self.enabled:
            return False, "disabled"
        
        if len(query) < self.min_length:
            return True, "too_short"
        
        if not any(kw in query for kw in self.DOMAIN_KEYWORDS):
            return True, "missing_domain_keyword"
        
        if any(p in query for p in self.COLLOQUIAL_PATTERNS):
            return True, "colloquial_expression"
        
        return False, ""
    
    def rewrite(self, query: str) -> RewriteResult:
        """
        Execute query rewrite.
        
        Args:
            query: Original query
            
        Returns:
            RewriteResult with rewritten query and metadata
        """
        should, reason = self.should_rewrite(query)
        
        if not should:
            return RewriteResult(query, 1.0, False, reason)
        
        if not self.llm:
            logger.warning("LLM wrapper not available, returning original query")
            return RewriteResult(query, 0.0, False, "no_llm")
        
        try:
            prompt = f"用户查询：{query}\n\n请改写此查询："
            result = self.llm.invoke(prompt, system=self.SYSTEM_PROMPT)
            
            rewritten = self._parse_result(result, query)
            
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}' (reason: {reason})")
            return RewriteResult(rewritten, 0.8, True, reason)
            
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}, returning original")
            return RewriteResult(query, 0.0, False, f"error: {e}")
    
    def _parse_result(self, result: str, original_query: str) -> str:
        """Parse LLM result to extract rewritten query."""
        try:
            json_match = re.search(r'\{[^}]+\}', result)
            if json_match:
                data = json.loads(json_match.group())
                rewritten = data.get("rewritten", original_query)
                if rewritten and len(rewritten) >= 2:
                    return rewritten
        except (json.JSONDecodeError, KeyError):
            pass
        
        if result and len(result.strip()) >= 2:
            return result.strip()
        
        return original_query
    
    def is_available(self) -> bool:
        """Check if rewriter is available."""
        return self.enabled and self.llm is not None
    
    def get_stats(self) -> dict:
        """Get rewriter statistics."""
        return {
            "enabled": self.enabled,
            "min_length": self.min_length,
            "llm_available": self.llm is not None,
        }
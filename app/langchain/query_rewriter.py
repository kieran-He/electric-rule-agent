"""
LLM-powered Query Rewriter for Enhanced Retrieval

Rewrites ambiguous/colloquial queries into more precise forms for better retrieval.
Uses intelligent triggering to avoid unnecessary LLM calls.
Also detects province codes from user query for multi-province support.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

from app.langchain.llm import MiniMaxLLMWrapper
from dataprocess.province_mapping import PROVINCE_ALIASES, PROVINCE_CODE_ALIASES

logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    rewritten_query: str
    confidence: float
    triggered: bool
    trigger_reason: str
    province_codes: Optional[List[str]] = None


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
    
    SYSTEM_PROMPT = """你是电力政策检索专家。分析用户查询并返回JSON结果。

任务：
1. 改写查询：补充领域关键词，去除口语化表达
2. 识别省份：提取用户提到的省份（中文或代码）

省份映射（中文→代码）：
山西=SX, 陕西=SN, 甘肃=GS, 山东=SD, 安徽=AH
北京=BJ, 江苏=JS, 浙江=ZJ, 广东=GD, 四川=SC
内蒙古=NM, 辽宁=LN, 吉林=JL, 黑龙江=HL
福建=FJ, 江西=JX, 河南=HA, 湖北=HB, 湖南=HN
广西=GX, 海南=HI, 重庆=CQ, 贵州=GZ, 云南=YN
西藏=XZ, 青海=QH, 宁夏=NX, 新疆=XJ, 天津=TJ
河北=HE, 上海=SH

输出JSON格式：
{
  "rewritten": "改写后的查询",
  "province_codes": ["SN"]
}

规则：
- province_codes: 提取的省份代码列表（大写）
- 未提及任何省份时返回空数组 []
- 用户提及多个省份时返回多个代码，如 ["SN", "GS"]

示例：
输入："山西的电力市场规则是什么"
输出：{"rewritten": "山西省电力市场规则细则", "province_codes": ["SX"]}

输入："陕西和甘肃的中长期交易有什么区别"
输出：{"rewritten": "陕西省与甘肃省中长期电力交易规则对比", "province_codes": ["SN", "GS"]}

输入："2026年交易时间表"
输出：{"rewritten": "2026年电力市场交易时间安排表", "province_codes": []}"""

    def __init__(
        self,
        llm_wrapper: Optional[MiniMaxLLMWrapper] = None,
        enabled: bool = True,
        always_rewrite: bool = True,
    ):
        self.llm = llm_wrapper
        self.enabled = enabled
        self.always_rewrite = always_rewrite
    
    def rewrite(self, query: str) -> RewriteResult:
        """
        Execute query rewrite (triggers LLM call when always_rewrite=True).
        
        Args:
            query: Original query
            
        Returns:
            RewriteResult with rewritten query, province codes, and metadata
        """
        if not self.enabled:
            fallback_codes = self._extract_provinces_fallback(query)
            return RewriteResult(query, 1.0, False, "disabled", fallback_codes)
        
        if not self.always_rewrite:
            fallback_codes = self._extract_provinces_fallback(query)
            return RewriteResult(query, 1.0, False, "skip_rewrite", fallback_codes)
        
        if not self.llm:
            logger.warning("LLM wrapper not available, returning original query")
            fallback_codes = self._extract_provinces_fallback(query)
            return RewriteResult(query, 0.0, False, "no_llm", fallback_codes)
        
        try:
            prompt = f"用户查询：{query}\n\n请分析并改写此查询："
            result = self.llm.invoke_text(prompt, system=self.SYSTEM_PROMPT)
            
            rewritten, province_codes = self._parse_result(result, query)
            
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}', provinces: {province_codes}")
            return RewriteResult(rewritten, 0.8, True, "llm_rewrite", province_codes)
            
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}, returning original")
            fallback_codes = self._extract_provinces_fallback(query)
            return RewriteResult(query, 0.0, False, f"error: {e}", fallback_codes)
    
    def _parse_result(self, result: str, original_query: str) -> Tuple[str, List[str]]:
        """Parse LLM result to extract rewritten query and province codes."""
        try:
            json_match = re.search(r'\{[^}]+\}', result)
            if json_match:
                data = json.loads(json_match.group())
                rewritten = data.get("rewritten", original_query)
                province_codes = data.get("province_codes", [])
                
                valid_codes = [c.upper() for c in province_codes if c.upper() in PROVINCE_CODE_ALIASES]
                
                if rewritten and len(rewritten) >= 2:
                    return rewritten, valid_codes
        except (json.JSONDecodeError, KeyError):
            pass
        
        fallback_codes = self._extract_provinces_fallback(original_query)
        
        if result and len(result.strip()) >= 2:
            return result.strip(), fallback_codes
        
        return original_query, fallback_codes
    
    def _extract_provinces_fallback(self, text: str) -> List[str]:
        """Fallback: extract provinces using keyword matching."""
        codes = []
        for alias, code in PROVINCE_ALIASES.items():
            if alias in text:
                codes.append(code)
        return codes
    
    def is_available(self) -> bool:
        """Check if rewriter is available."""
        return self.enabled and self.llm is not None
    
    def should_rewrite(self, query: str) -> Tuple[bool, str]:
        """
        Legacy method - always returns True when enabled.
        Kept for backward compatibility.
        """
        if not self.enabled:
            return False, "disabled"
        return True, "always_rewrite"
    
    def get_stats(self) -> dict:
        """Get rewriter statistics."""
        return {
            "enabled": self.enabled,
            "always_rewrite": self.always_rewrite,
            "llm_available": self.llm is not None,
        }
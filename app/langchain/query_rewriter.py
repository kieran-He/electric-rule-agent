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
class QueryPlan:
    """Single query plan for retrieval."""
    query: str
    province_codes: List[str]


@dataclass
class RewriteResult:
    """Rewrite result supporting multi-query splitting."""
    queries: List[QueryPlan]
    should_split: bool
    split_reason: str
    triggered: bool
    trigger_reason: str
    
    @property
    def rewritten_query(self) -> str:
        """Return first query (backward compatible)."""
        return self.queries[0].query if self.queries else ""
    
    @property
    def province_codes(self) -> List[str]:
        """Return first query's provinces (backward compatible)."""
        return self.queries[0].province_codes if self.queries else []


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
3. 拆分查询：如果查询涉及多个省份或多个市场类型，拆分为独立的子查询

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
  "queries": [
    {"query": "改写后的查询1", "province_codes": ["SD"]},
    {"query": "改写后的查询2", "province_codes": ["HI"]}
  ],
  "should_split": true,
  "split_reason": "拆分原因（如：多省份查询需分别检索）"
}

拆分规则：
1. 多省份查询：拆分为每个省份的独立查询
2. 单省份查询：返回单个查询对象
3. 无省份查询：province_codes为空数组 []

示例：
输入："山东海南中长期电力市场交易对发电量的要求"
输出：{
  "queries": [
    {"query": "山东省中长期电力市场交易规则对发电量的要求", "province_codes": ["SD"]},
    {"query": "海南省中长期电力市场交易规则对发电量的要求", "province_codes": ["HI"]}
  ],
  "should_split": true,
  "split_reason": "涉及山东和海南两省份，需分别检索"
}

输入："陕西现货市场交易时间"
输出：{
  "queries": [
    {"query": "陕西省现货市场交易时间安排", "province_codes": ["SN"]}
  ],
  "should_split": false,
  "split_reason": "单省份查询无需拆分"
}"""

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
        Execute query rewrite with splitting support.
        
        Args:
            query: Original query
            
        Returns:
            RewriteResult with queries, should_split, and metadata
        """
        if not self.enabled or not self.always_rewrite or not self.llm:
            fallback_codes = self._extract_provinces_fallback(query)
            return RewriteResult(
                queries=[QueryPlan(query, fallback_codes)],
                should_split=False,
                split_reason="disabled_or_no_llm",
                triggered=False,
                trigger_reason="disabled"
            )
        
        try:
            prompt = f"用户查询：{query}\n\n请分析、改写并拆分此查询："
            result = self.llm.invoke_text(prompt, system=self.SYSTEM_PROMPT)
            
            queries, should_split, reason = self._parse_result(result, query)
            
            logger.info(f"Query rewritten: '{query}' -> {len(queries)} queries, should_split={should_split}")
            for i, qp in enumerate(queries):
                logger.info(f"  Query {i+1}: '{qp.query}', provinces: {qp.province_codes}")
            
            return RewriteResult(
                queries=queries,
                should_split=should_split,
                split_reason=reason,
                triggered=True,
                trigger_reason="llm_rewrite_and_split"
            )
            
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            fallback_codes = self._extract_provinces_fallback(query)
            return RewriteResult(
                queries=[QueryPlan(query, fallback_codes)],
                should_split=False,
                split_reason=f"error: {e}",
                triggered=False,
                trigger_reason="error"
            )
    
    def _parse_result(self, result: str, original_query: str) -> Tuple[List[QueryPlan], bool, str]:
        """Parse LLM result to extract query plans."""
        try:
            json_str = self._extract_json(result)
            if json_str:
                data = json.loads(json_str)
                
                queries_data = data.get("queries", [])
                should_split = data.get("should_split", False)
                reason = data.get("split_reason", "")
                
                query_plans = []
                for qd in queries_data:
                    q_text = qd.get("query", original_query)
                    q_codes = qd.get("province_codes", [])
                    valid_codes = [c.upper() for c in q_codes if c.upper() in PROVINCE_CODE_ALIASES]
                    
                    if q_text and len(q_text.strip()) >= 2:
                        query_plans.append(QueryPlan(q_text.strip(), valid_codes))
                
                if query_plans:
                    return query_plans, should_split, reason
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"JSON parse failed: {e}, result snippet: {result[:200]}")
        
        fallback_codes = self._extract_provinces_fallback(original_query)
        return [QueryPlan(original_query, fallback_codes)], False, "parse_failed"
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text by finding balanced braces."""
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        brace_count = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if brace_count == 0 and end_idx > start_idx:
            return text[start_idx:end_idx]
        
        return None
    
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
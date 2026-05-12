"""
LLM-powered Query Rewriter with Coreference Resolution

Rewrites ambiguous/colloquial queries into more precise forms for better retrieval.
Resolves pronouns and references using conversation history.
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
    LLM-powered query rewriter with coreference resolution.
    
    Features:
    - Resolves pronouns/references from conversation history
    - Rewrites ambiguous/colloquial queries
    - Detects and splits multi-province queries
    - Compresses conversation history for context
    """
    
    DOMAIN_KEYWORDS = [
        "电力", "交易", "结算", "现货", "中长期", "零售", 
        "辅助", "储能", "调频", "市场", "规则", "政策",
        "电价", "负荷", "发电", "用电", "电网"
    ]
    
    SYSTEM_PROMPT = """你是电力政策检索专家。分析用户查询并返回JSON结果。

任务：
1. **指代消解**：根据对话历史，将代词（它、那个、这个）替换为具体政策名称或文档名称
2. **查询改写**：补充领域关键词，去除口语化表达
3. **省份识别**：提取用户提到的省份（中文或代码）
4. **查询拆分**：如果查询涉及多个省份或多个市场类型，拆分为独立的子查询

指代消解示例：
- "它怎么规定的" → "陕西电力市场结算细则怎么规定的"
- "那个政策适用范围" → "陕西省电力中长期市场实施细则适用范围"
- "刚才说的文件" → "陕西电力现货市场交易实施细则"

省份映射（中文→代码）：
陕西=SN, 甘肃=GS, 山西=SX, 山东=SD, 安徽=AH
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
  "split_reason": "拆分原因"
}

拆分规则：
1. 多省份查询：拆分为每个省份的独立查询
2. 单省份查询：返回单个查询对象
3. 无省份查询：province_codes为空数组 []"""

    def __init__(
        self,
        llm_wrapper: Optional[MiniMaxLLMWrapper] = None,
        enabled: bool = True,
        always_rewrite: bool = True,
    ):
        self.llm = llm_wrapper
        self.enabled = enabled
        self.always_rewrite = always_rewrite
    
    def rewrite(self, query: str, history: List[str] = None) -> RewriteResult:
        """
        Execute query rewrite with coreference resolution.
        
        Args:
            query: Original query
            history: Conversation history list (optional)
            
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
            compressed_history = self._compress_history(history) if history else ""
            
            if compressed_history:
                prompt = f"对话历史：\n{compressed_history}\n\n当前问题：{query}\n\n请分析当前问题，结合对话历史进行指代消解、改写并拆分："
            else:
                prompt = f"当前问题：{query}\n\n请分析、改写并拆分此问题："
            
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
                trigger_reason="llm_rewrite_with_coreference"
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
    
    def _compress_history(self, history: List[str], max_turns: int = 6, max_chars: int = 800) -> str:
        """
        Compress conversation history for LLM context.
        
        Args:
            history: Full conversation history
            max_turns: Maximum number of recent turns to include
            max_chars: Maximum total characters
            
        Returns:
            Compressed history string
        """
        if not history:
            return ""
        
        recent = history[-max_turns:] if len(history) > max_turns else history
        
        compressed_lines = []
        total_chars = 0
        
        for turn in recent:
            clean_turn = self._clean_turn_text(turn)
            
            if len(clean_turn) > 150:
                entities = self._extract_entities_from_turn(clean_turn)
                if entities:
                    compressed_turn = f"提到: {entities}"
                else:
                    compressed_turn = clean_turn[:150] + "..."
            else:
                compressed_turn = clean_turn
            
            if total_chars + len(compressed_turn) > max_chars:
                break
            
            compressed_lines.append(compressed_turn)
            total_chars += len(compressed_turn)
        
        return "\n".join(compressed_lines)
    
    def _clean_turn_text(self, turn: str) -> str:
        """Remove Q:/A: prefixes from turn text."""
        if turn.startswith("Q: ") or turn.startswith("A: "):
            return turn[3:]
        if turn.startswith("【历史摘要】"):
            return turn[6:]
        return turn
    
    def _extract_entities_from_turn(self, text: str) -> str:
        """Extract key entities (policy name, province) from text."""
        entities = []
        
        policy_patterns = [
            r"《([^《》]+?细则)》",
            r"《([^《》]+?规则)》",
            r"《([^《》]+?办法)》",
        ]
        for pattern in policy_patterns:
            match = re.search(pattern, text)
            if match:
                entities.append(match.group(0))
                break
        
        for alias in sorted(PROVINCE_ALIASES.keys(), key=len, reverse=True):
            if alias in text:
                entities.append(alias)
                break
        
        return ", ".join(entities) if entities else ""
    
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
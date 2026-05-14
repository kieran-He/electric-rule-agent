from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.langchain.llm import MiniMaxLLMWrapper

logger = logging.getLogger(__name__)


INTENT_CLASSIFICATION_PROMPT = """你是一个意图分类专家。请分析用户查询，返回JSON格式的分类结果。

用户查询：{query}

规则分类提示：{rule_hint}

请返回以下JSON格式（不要添加markdown代码块标记，直接返回JSON）：
{{
  "intent": "policy_query 或 data_query 或 analysis 或 hybrid",
  "sub_intents": ["如果有多个意图，列出子意图"],
  "confidence": 0.0到1.0之间的置信度,
  "reason": "分类理由的简要说明",
  "suggested_plan": [
    {{"step": 1, "action": "建议的执行步骤1"}},
    {{"step": 2, "action": "建议的执行步骤2"}}
  ]
}}

意图定义：
- policy_query: 用户询问政策、规则、通知等相关内容
- data_query: 用户查询电力数据、负荷、发电量等数值数据
- analysis: 用户需要统计分析、趋势预测、数据对比等分析操作
- hybrid: 用户请求包含多种意图，需要综合处理"""


class LLMClassifier:
    @classmethod
    def classify(
        cls,
        query: str,
        rule_result: Dict,
        llm_wrapper: Optional["MiniMaxLLMWrapper"] = None,
    ) -> Dict:
        if not llm_wrapper:
            logger.warning("[LLMClassifier] No LLM wrapper available, using rule result")
            return cls._fallback_result(query, rule_result)
        
        rule_hint = f"意图: {rule_result.get('intent', 'unknown')}, 置信度: {rule_result.get('confidence', 0):.2f}, 理由: {rule_result.get('reason', '')}"
        
        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            query=query,
            rule_hint=rule_hint,
        )
        
        try:
            response = llm_wrapper.invoke_text(query, system=prompt)
            logger.info(f"[LLMClassifier] Raw response: {response[:200]}")
            
            result = cls._parse_response(response)
            result = cls._validate_result(result, rule_result)
            
            return result
            
        except Exception as e:
            logger.exception(f"[LLMClassifier] LLM classification failed: {e}")
            return cls._fallback_result(query, rule_result)
    
    @classmethod
    def _parse_response(cls, response: str) -> Dict:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            logger.warning(f"[LLMClassifier] No JSON found in response: {response[:100]}")
            return cls._default_result()
        
        json_str = json_match.group()
        
        try:
            result = json.loads(json_str)
            
            return {
                "intent": result.get("intent", "hybrid"),
                "sub_intents": result.get("sub_intents", []),
                "confidence": float(result.get("confidence", 0.7)),
                "reason": result.get("reason", ""),
                "suggested_plan": result.get("suggested_plan", []),
                "detected_regions": [],
            }
        except json.JSONDecodeError as e:
            logger.warning(f"[LLMClassifier] JSON parse error: {e}")
            return cls._default_result()
    
    @classmethod
    def _validate_result(cls, result: Dict, rule_result: Dict) -> Dict:
        valid_intents = ["policy_query", "data_query", "analysis", "hybrid"]
        if result.get("intent") not in valid_intents:
            result["intent"] = rule_result.get("intent", "hybrid")
        
        confidence = result.get("confidence", 0.7)
        if not isinstance(confidence, (int, float)):
            confidence = 0.7
        result["confidence"] = max(0.0, min(1.0, float(confidence)))
        
        result["detected_regions"] = rule_result.get("detected_regions", [])
        
        return result
    
    @classmethod
    def _default_result(cls) -> Dict:
        return {
            "intent": "hybrid",
            "sub_intents": [],
            "confidence": 0.5,
            "reason": "默认分类结果",
            "suggested_plan": cls._default_plan("hybrid"),
            "detected_regions": [],
        }
    
    @classmethod
    def _default_plan(cls, intent: str) -> List[Dict]:
        plans = {
            "policy_query": [
                {"step": 1, "action": "检索相关政策文档"},
                {"step": 2, "action": "生成政策解答"},
            ],
            "data_query": [
                {"step": 1, "action": "获取电力数据"},
                {"step": 2, "action": "分析数据并生成报告"},
            ],
            "analysis": [
                {"step": 1, "action": "获取历史数据"},
                {"step": 2, "action": "执行统计分析"},
                {"step": 3, "action": "生成分析报告"},
            ],
            "hybrid": [
                {"step": 1, "action": "检索相关政策文档"},
                {"step": 2, "action": "获取相关数据"},
                {"step": 3, "action": "综合分析并生成回答"},
            ],
        }
        return plans.get(intent, plans["hybrid"])
    
    @classmethod
    def _fallback_result(cls, query: str, rule_result: Dict) -> Dict:
        return {
            "intent": rule_result.get("intent", "hybrid"),
            "sub_intents": rule_result.get("sub_intents", []),
            "confidence": rule_result.get("confidence", 0.5),
            "reason": f"LLM不可用，使用规则分类: {rule_result.get('reason', '')}",
            "suggested_plan": cls._default_plan(rule_result.get("intent", "hybrid")),
            "detected_regions": rule_result.get("detected_regions", []),
        }
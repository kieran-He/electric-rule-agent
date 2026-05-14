from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class RuleClassifier:
    # Policy context patterns - when these appear, treat as policy even if data keywords present
    POLICY_CONTEXT_PATTERNS = [
        "如何计算", "怎么计算", "计算方法", "计算规则",
        "关系是什么", "有什么关系", "之间的关系",
        "是什么", "定义", "含义", "概念",
        "规则", "规定", "办法", "机制", "流程",
    ]
    
    POLICY_KEYWORDS = {
        "primary": ["政策", "规则", "通知", "规定", "条款", "准入", "交易规则", "管理办法", "实施细则"],
        "secondary": ["文件", "要求", "条件", "标准", "办法", "条例", "法规"],
    }
    
    # Data keywords that require actual numerical data retrieval
    DATA_KEYWORDS = {
        "primary": ["负荷", "发电量", "用电量", "曲线", "发电", "用电"],
        "secondary": ["数值", "统计", "记录", "历史", "当前", "查询", "获取", "数据"],
    }
    
    # Price-related: only data query if asking for actual values
    DATA_PRICE_PATTERNS = [
        "电价数据", "实时电价", "当前电价", "历史电价", 
        "昨天电价", "今日电价", "电价是多少", "电价数值",
    ]
    
    ANALYSIS_KEYWORDS = {
        "primary": ["分析", "统计", "均值", "方差", "趋势", "增长", "分布", "对比", "比较"],
        "secondary": ["计算", "评估", "预测", "变化", "波动", "异常"],
    }
    
    REGION_KEYWORDS = {
        "陕西": "SN", "陕西省": "SN", "shanxi": "SN",
        "山西": "SX", "山西省": "SX",
        "甘肃": "GS", "甘肃省": "GS",
        "山东": "SD", "山东省": "SD",
        "安徽": "AH", "安徽省": "AH",
        "广东": "GD", "广东省": "GD",
        "浙江": "ZJ", "浙江省": "ZJ",
    }
    
    @classmethod
    def classify(cls, query: str) -> Dict:
        scores = cls._score_intent(query)
        confidence = cls._calculate_confidence(scores)
        regions = cls._detect_regions(query)
        
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_score = sorted_intents[0]
        second_intent, second_score = sorted_intents[1] if len(sorted_intents) > 1 else (None, 0)
        
        if top_score == 0:
            return {
                "intent": "hybrid",
                "confidence": 0.5,
                "reason": "未检测到明确意图关键词",
                "detected_regions": regions,
                "sub_intents": [],
            }
        
        if top_score > 0 and second_score > 0:
            score_ratio = top_score / (top_score + second_score)
            if score_ratio < 0.7:
                return {
                    "intent": "hybrid",
                    "confidence": 0.6,
                    "reason": f"检测到多个意图：{top_intent}({top_score:.2f}) + {second_intent}({second_score:.2f})",
                    "detected_regions": regions,
                    "sub_intents": [top_intent, second_intent],
                }
        
        intent_map = {
            "policy": "policy_query",
            "data": "data_query",
            "analysis": "analysis",
        }
        
        final_intent = intent_map.get(top_intent, "hybrid")
        
        return {
            "intent": final_intent,
            "confidence": confidence,
            "reason": f"规则分类：{top_intent}得分最高({top_score:.2f})",
            "detected_regions": regions,
            "sub_intents": [],
        }
    
    @classmethod
    def _score_intent(cls, query: str) -> Dict[str, float]:
        scores = {"policy": 0.0, "data": 0.0, "analysis": 0.0}
        
        # Check for policy context patterns first - these override data keywords
        has_policy_context = False
        for pattern in cls.POLICY_CONTEXT_PATTERNS:
            if pattern in query:
                has_policy_context = True
                scores["policy"] += 3.0  # Strong policy signal
                break
        
        # Policy keywords
        for kw in cls.POLICY_KEYWORDS["primary"]:
            if kw in query:
                scores["policy"] += 2.0
        for kw in cls.POLICY_KEYWORDS["secondary"]:
            if kw in query:
                scores["policy"] += 0.5
        
        # Data keywords - only score if not asking about rules/methods
        if not has_policy_context:
            for kw in cls.DATA_KEYWORDS["primary"]:
                if kw in query:
                    scores["data"] += 2.0
            for kw in cls.DATA_KEYWORDS["secondary"]:
                if kw in query:
                    scores["data"] += 0.5
            
            # Price patterns - only data if asking for actual values
            for pattern in cls.DATA_PRICE_PATTERNS:
                if pattern in query:
                    scores["data"] += 2.0
        
        # Analysis keywords
        for kw in cls.ANALYSIS_KEYWORDS["primary"]:
            if kw in query:
                scores["analysis"] += 2.0
        for kw in cls.ANALYSIS_KEYWORDS["secondary"]:
            if kw in query:
                scores["analysis"] += 0.5
        
        return scores
    
    @classmethod
    def _calculate_confidence(cls, scores: Dict[str, float]) -> float:
        total = sum(scores.values())
        if total == 0:
            return 0.3
        
        sorted_scores = sorted(scores.values(), reverse=True)
        top_score = sorted_scores[0]
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0
        
        if second_score == 0:
            ratio = min(top_score / 4.0, 1.0)
            return 0.7 + 0.25 * ratio
        
        if top_score > second_score * 2:
            return 0.85
        elif top_score > second_score * 1.5:
            return 0.75
        else:
            return 0.6
    
    @classmethod
    def _detect_regions(cls, query: str) -> List[str]:
        regions = []
        for name, code in cls.REGION_KEYWORDS.items():
            if name in query:
                if code not in regions:
                    regions.append(code)
        return regions
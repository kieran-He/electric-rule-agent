"""
Coreference Resolution for Multi-turn Conversations

Resolves pronouns and references in queries to specific entities from conversation history.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from dataprocess.province_mapping import PROVINCE_ALIASES

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntities:
    policy_name: Optional[str] = None
    doc_name: Optional[str] = None
    province: Optional[str] = None
    topic: Optional[str] = None


class CoreferenceResolver:
    """
    Multi-turn conversation coreference resolver.
    
    Resolves references like "那个政策", "它", "那个文件" to specific entities
    extracted from conversation history.
    """
    
    COREFERENCE_INDICATORS = [
        "那个政策", "这个政策", "那个规则", "这个规则",
        "那个文件", "这个文件", "那个文档", "这个文档",
        "那个省份", "这个省份", "那个省", "这个省",
        "刚才说的", "上面说的", "之前说的",
        "它", "其",
    ]
    
    POLICY_PATTERNS = [
        r"《([^《》]+?规则)》",
        r"《([^《》]+?细则)》",
        r"《([^《》]+?办法)》",
        r"《([^《》]+?规定)》",
        r"《([^《》]+?通知)》",
        r"([^《》\s]+?电力市场交易规则)",
        r"([^《》\s]+?现货市场交易规则)",
    ]
    
    DOC_PATTERNS = [
        r"《([^《》]+?\.pdf)》",
        r"《([^《》]+?\.docx?)》",
        r"参考[《]?([^《》\s]+?\.pdf)[》]?",
        r"下载[《]?([^《》\s]+?\.pdf)[》]?",
    ]
    
    TOPIC_KEYWORDS = [
        "现货市场", "中长期交易", "零售市场", "辅助服务",
        "储能", "调频", "结算", "电价", "交易时间",
        "市场主体", "发电企业", "售电公司", "用户",
    ]
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    def resolve(self, query: str, history: List[str]) -> str:
        """
        Resolve coreferences in query using conversation history.
        
        Args:
            query: The user's current query
            history: List of conversation turns ["Q: xxx", "A: xxx", ...]
            
        Returns:
            Resolved query with coreferences replaced by specific entities
        """
        if not self.enabled:
            return query
        
        if not self._has_coreference(query):
            return query
        
        entities = self._extract_entities(history)
        
        if not any([entities.policy_name, entities.doc_name, entities.province, entities.topic]):
            logger.debug("No entities found in history for coreference resolution")
            return query
        
        resolved = self._replace_coreference(query, entities)
        
        if resolved != query:
            logger.info(f"Coreference resolved: '{query}' -> '{resolved}'")
        
        return resolved
    
    def _has_coreference(self, query: str) -> bool:
        """Check if query contains coreference indicators."""
        for indicator in self.COREFERENCE_INDICATORS:
            if indicator in query:
                return True
        return False
    
    def _extract_entities(self, history: List[str]) -> ExtractedEntities:
        """
        Extract key entities from conversation history.
        
        Searches from most recent to oldest within last 6 turns.
        """
        entities = ExtractedEntities()
        
        if not history:
            return entities
        
        recent_history = history[-6:] if len(history) > 6 else history
        
        for turn in reversed(recent_history):
            text = self._clean_turn_text(turn)
            
            if not entities.policy_name:
                entities.policy_name = self._extract_policy_name(text)
            
            if not entities.doc_name:
                entities.doc_name = self._extract_doc_name(text)
            
            if not entities.province:
                entities.province = self._extract_province(text)
            
            if not entities.topic:
                entities.topic = self._extract_topic(text)
        
        return entities
    
    def _clean_turn_text(self, turn: str) -> str:
        """Remove Q:/A: prefixes from turn text."""
        if turn.startswith("Q: ") or turn.startswith("A: "):
            return turn[3:]
        if turn.startswith("【历史摘要】"):
            return turn[6:]
        return turn
    
    def _extract_policy_name(self, text: str) -> Optional[str]:
        """Extract policy name using regex patterns."""
        for pattern in self.POLICY_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _extract_doc_name(self, text: str) -> Optional[str]:
        """Extract document name using regex patterns."""
        for pattern in self.DOC_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _extract_province(self, text: str) -> Optional[str]:
        """Extract province name from text."""
        for alias in sorted(PROVINCE_ALIASES.keys(), key=len, reverse=True):
            if alias in text:
                return alias
        return None
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """Extract topic keyword from text."""
        for topic in self.TOPIC_KEYWORDS:
            if topic in text:
                return topic
        return None
    
    def _replace_coreference(self, query: str, entities: ExtractedEntities) -> str:
        """Replace coreference indicators with specific entities."""
        result = query
        
        if entities.policy_name:
            result = result.replace("那个政策", entities.policy_name)
            result = result.replace("这个政策", entities.policy_name)
            result = result.replace("那个规则", entities.policy_name)
            result = result.replace("这个规则", entities.policy_name)
        
        if entities.doc_name:
            result = result.replace("那个文件", f"《{entities.doc_name}》")
            result = result.replace("这个文件", f"《{entities.doc_name}》")
            result = result.replace("那个文档", f"《{entities.doc_name}》")
            result = result.replace("这个文档", f"《{entities.doc_name}》")
        
        if entities.province:
            result = result.replace("那个省份", entities.province)
            result = result.replace("这个省份", entities.province)
            result = result.replace("那个省", entities.province)
            result = result.replace("这个省", entities.province)
        
        result = self._replace_pronouns(result, entities)
        
        result = self._replace_contextual_references(result, entities)
        
        return result
    
    def _replace_pronouns(self, query: str, entities: ExtractedEntities) -> str:
        """Replace single-character pronouns (它, 其) based on context."""
        if "它" not in query and "其" not in query:
            return query
        
        replacement = None
        
        context_keywords = ["适用", "范围", "规定", "要求", "内容", "条款", "实施"]
        if any(kw in query for kw in context_keywords):
            replacement = entities.policy_name or entities.topic
        
        if replacement:
            query = query.replace("它", replacement)
            query = query.replace("其", replacement)
        
        return query
    
    def _replace_contextual_references(self, query: str, entities: ExtractedEntities) -> str:
        """Replace contextual references like '刚才说的'."""
        if "刚才说的" in query or "上面说的" in query or "之前说的" in query:
            context = entities.policy_name or entities.topic
            if context:
                query = query.replace("刚才说的", context)
                query = query.replace("上面说的", context)
                query = query.replace("之前说的", context)
        
        return query
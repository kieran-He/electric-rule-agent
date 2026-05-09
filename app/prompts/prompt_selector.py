"""
Prompt 选择器：根据意图自动选择合适的 Prompt 模板
"""
from typing import Tuple

from app.prompts.prompt_templates import (
    QA_STRUCTURED_PROMPT,
    COMPARE_STRUCTURED_PROMPT,
    EXPLAIN_STRUCTURED_PROMPT,
    PROCEDURE_STRUCTURED_PROMPT,
    INTENT_KEYWORD_MAP,
)


class PromptSelector:
    """
    智能选择 Prompt 模板（规则优先，关键词匹配）
    """
    
    def __init__(self):
        self.keyword_map = INTENT_KEYWORD_MAP
    
    def select_prompt(self, query: str) -> Tuple[str, str]:
        """
        选择合适的 Prompt 模板
        
        Args:
            query: 用户查询
            
        Returns:
            (system_prompt, intent)
        """
        intent = self._classify_by_rules(query)
        
        prompt_map = {
            "query": QA_STRUCTURED_PROMPT,
            "compare": COMPARE_STRUCTURED_PROMPT,
            "explain": EXPLAIN_STRUCTURED_PROMPT,
            "procedure": PROCEDURE_STRUCTURED_PROMPT,
        }
        
        return prompt_map.get(intent, QA_STRUCTURED_PROMPT), intent
    
    def _classify_by_rules(self, query: str) -> str:
        """规则优先的意图分类（关键词匹配）"""
        for intent, keywords in self.keyword_map.items():
            for kw in keywords:
                if kw in query:
                    return intent
        return "query"
    
    def get_intent(self, query: str) -> str:
        """获取意图分类结果"""
        return self._classify_by_rules(query)
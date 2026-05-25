from __future__ import annotations

import logging
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.langchain.llm import MiniMaxLLMWrapper

logger = logging.getLogger(__name__)


class TitleGenerator:
    
    def __init__(
        self,
        llm_wrapper: Optional["MiniMaxLLMWrapper"] = None,
        max_title_length: int = 15,
    ):
        self.llm = llm_wrapper
        self.max_title_length = max_title_length
    
    def is_available(self) -> bool:
        return self.llm is not None
    
    def generate(self, history: List[str]) -> str:
        if not history or len(history) < 2:
            logger.warning("[TitleGenerator] Insufficient history for title generation")
            return "新对话"
        
        if not self.is_available():
            logger.warning("[TitleGenerator] LLM not available, using fallback")
            return self._fallback_title(history)
        
        first_question = ""
        for item in history:
            if item.startswith("Q: "):
                first_question = item[3:].strip()
                break
        
        if not first_question:
            return self._fallback_title(history)
        
        try:
            title = self._generate_with_llm(first_question)
            return title
        except Exception as e:
            logger.exception(f"[TitleGenerator] Failed to generate title: {e}")
            return self._fallback_title(history)
    
    def _generate_with_llm(self, question: str) -> str:
        system_prompt = f"""你是一个标题生成助手。根据用户的第一个问题生成一个简短的对话标题。

规则：
1. 标题长度严格控制在{self.max_title_length}字以内
2. 直接输出标题，不要任何标点符号或解释
3. 提炼问题核心，不要包含"请问"、"我想了解"等冗余词
4. 保持标题简洁、专业"""

        user_content = f"用户问题：{question}\n\n请生成标题："
        
        response, _, _ = self.llm.invoke(user_content, system=system_prompt)
        
        title = response.strip()
        title = title.replace('"', '').replace('"', '').replace('"', '')
        title = title.replace('「', '').replace('」', '')
        
        if len(title) > self.max_title_length:
            title = title[:self.max_title_length]
        
        return title
    
    def _fallback_title(self, history: List[str]) -> str:
        for item in history:
            if item.startswith("Q: "):
                question = item[3:].strip()
                if len(question) <= self.max_title_length:
                    return question
                return question[:self.max_title_length]
        
        return "新对话"
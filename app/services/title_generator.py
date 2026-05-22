"""
Title Generator for Conversation Sessions

Uses LLM to generate concise titles from conversation content.
"""
from __future__ import annotations

import logging
from typing import List

from app.langchain.llm import MiniMaxLLMWrapper

logger = logging.getLogger(__name__)


class TitleGenerator:
    """
    LLM-powered title generator for conversation sessions.
    
    Generates concise titles (≤20 characters) from first conversation turn.
    """
    
    SYSTEM_PROMPT = "你是对话标题生成助手。请根据用户问题生成一个简短的标题，不超过20个字。只返回标题文本，不要添加任何标点符号、引号或解释。"

    def __init__(
        self,
        llm_wrapper: MiniMaxLLMWrapper = None,
        max_title_length: int = 20,
    ):
        self.llm = llm_wrapper or MiniMaxLLMWrapper()
        self.max_title_length = max_title_length
    
    def generate(self, history: List[str]) -> str:
        """
        Generate title from conversation history.
        
        Args:
            history: List of conversation entries in format ["Q: xxx", "A: xxx", ...]
            
        Returns:
            Generated title (max 20 characters)
        """
        if not history:
            return "新对话"
        
        user_query = None
        bot_reply = None
        
        for i, entry in enumerate(history):
            if entry.startswith("Q: ") and user_query is None:
                user_query = entry[3:]
            elif entry.startswith("A: ") and bot_reply is None:
                bot_reply = entry[3:]
            if user_query and bot_reply:
                break
        
        if user_query is None:
            if history and history[0].startswith("【历史摘要】"):
                summary = history[0][6:]
                return summary[:self.max_title_length]
            return "新对话"
        
        try:
            prompt = self._build_prompt(user_query, bot_reply)
            
            title = self.llm.invoke_text(prompt, system=self.SYSTEM_PROMPT)
            
            title = title.strip().strip('"\'""''「」【】')
            
            if len(title) > self.max_title_length:
                title = title[:self.max_title_length]
            
            logger.debug(f"Title generated: {title}")
            
            return title if title else "新对话"
            
        except Exception as e:
            logger.warning(f"Title generation failed: {e}, using fallback")
            
            return user_query[:self.max_title_length] if user_query else "新对话"
    
    def _build_prompt(self, user_query: str, bot_reply: str | None) -> str:
        """Build prompt for title generation."""
        if bot_reply:
            reply_snippet = bot_reply[:100] if len(bot_reply) > 100 else bot_reply
            return f"用户问题：{user_query}\n回答摘要：{reply_snippet}\n\n请生成一个简短的标题（不超过20字）："
        return f"用户问题：{user_query}\n\n请生成一个简短的标题（不超过20字）："
    
    def is_available(self) -> bool:
        """Check if title generator is available."""
        return self.llm is not None
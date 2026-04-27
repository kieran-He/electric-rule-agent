"""
History Summarizer for Conversation Compression

Uses LLM to compress long conversation history into concise summaries.
"""
from __future__ import annotations

import logging
from typing import List

from app.langchain.llm import MiniMaxLLMWrapper
from app.db.models.conversation_turn import ConversationTurn

logger = logging.getLogger(__name__)


class HistorySummarizer:
    """
    LLM-powered history summarizer for conversation compression.
    
    Compresses multiple conversation turns into a concise summary
    to reduce token consumption while preserving key information.
    """
    
    SYSTEM_PROMPT = """你是对话历史压缩助手。请将以下对话历史压缩为简洁的摘要，遵循以下规则：

1. 保留关键信息：用户意图、关注省份、主要问题
2. 去除冗余细节：具体答案内容、次要信息
3. 使用简洁语言：不超过200字
4. 格式化输出：按时间顺序描述用户关注点变化

示例输出：
"用户询问陕西电力交易规则，后询问具体流程，关注中长期交易操作步骤。"""

    def __init__(
        self,
        llm_wrapper: MiniMaxLLMWrapper = None,
        max_summary_length: int = 200,
    ):
        self.llm = llm_wrapper or MiniMaxLLMWrapper()
        self.max_summary_length = max_summary_length
    
    def summarize(self, turns: List[ConversationTurn]) -> str:
        """
        Compress conversation turns into summary.
        
        Args:
            turns: List of ConversationTurn to compress
            
        Returns:
            Summary text (max 200 characters)
        """
        if not turns:
            return ""
        
        # Build conversation text
        conversation_lines = []
        for turn in turns:
            # Truncate bot_reply to avoid long context
            bot_reply_snippet = turn.bot_reply[:100] if len(turn.bot_reply) > 100 else turn.bot_reply
            conversation_lines.append(f"用户：{turn.user_query}")
            conversation_lines.append(f"系统：{bot_reply_snippet}")
        
        conversation_text = "\n".join(conversation_lines)
        
        # Call LLM to generate summary
        try:
            prompt = f"对话历史：\n{conversation_text}\n\n请压缩为不超过{self.max_summary_length}字的摘要："
            
            summary = self.llm.invoke_text(prompt, system=self.SYSTEM_PROMPT)
            
            # Truncate if exceeds max length
            if len(summary) > self.max_summary_length:
                summary = summary[:self.max_summary_length]
            
            logger.debug(f"History summarized: {len(turns)} turns -> {len(summary)} chars")
            
            return summary
            
        except Exception as e:
            logger.warning(f"History summarization failed: {e}, using fallback summary")
            
            # Fallback: simple concatenation of user queries
            queries = [t.user_query[:30] for t in turns[:5]]
            return f"用户询问：{', '.join(queries)}"
    
    def is_available(self) -> bool:
        """Check if summarizer is available."""
        return self.llm is not None
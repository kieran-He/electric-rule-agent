from __future__ import annotations

import logging
from typing import Any, Dict

from app.agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class GeneralChatTool(BaseTool):
    name = "general_chat"
    description = "通用对话工具，用于处理非电力领域的一般对话。当其他工具无法回答时作为备用。"
    keywords = []
    
    def __init__(self, llm_wrapper: Any):
        super().__init__()
        self._llm_wrapper = llm_wrapper
    
    def is_applicable(self, query: str) -> bool:
        return False
    
    def execute(self, query: str, context: Dict[str, Any] = None) -> ToolResult:
        ctx = context or self._context
        history = ctx.get("history", [])
        
        system_prompt = """你是电力政策问答助手。

当用户的问题不属于电力领域时，你可以友好地回答一般性问题，但应引导用户回到电力政策话题。

回答要求：
1. 对非电力问题进行友好回答
2. 简要提示你的主要职能是电力政策问答
3. 保持简洁，不超过100字"""
        
        history_text = ""
        if history:
            recent_history = history[-6:]
            history_text = "\n".join(recent_history)
        
        user_content = f"""历史对话：
{history_text}

当前问题：{query}

请回答用户的问题。"""
        
        try:
            answer, input_tokens, output_tokens = self._llm_wrapper.invoke(
                user_content, system=system_prompt
            )
            
            return ToolResult(
                success=True,
                output=answer,
                metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                tool_name=self.name,
                confidence=0.3,
            )
        except Exception as e:
            logger.exception(f"GeneralChatTool execution failed: {e}")
            return ToolResult(
                success=False,
                output="抱歉，我暂时无法回答这个问题。",
                tool_name=self.name,
                confidence=0.0,
            )
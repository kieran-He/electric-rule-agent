from __future__ import annotations

import logging
from typing import Any, Dict, List, TYPE_CHECKING

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.prompts import REACT_SYSTEM_PROMPT
from app.agent.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from app.langchain.llm import MiniMaxLLMWrapper

logger = logging.getLogger(__name__)


class ReActAgent:
    """
    LangChain Agent wrapper for multi-tool orchestration.
    
    Uses langchain.agents.create_agent() for tool calling loop.
    """
    
    def __init__(
        self,
        llm_wrapper: "MiniMaxLLMWrapper",
        tools: List[BaseTool],
        max_iterations: int = 3,
    ):
        self._llm_wrapper = llm_wrapper
        self._llm = llm_wrapper._get_client()
        self._tools = tools
        self._max_iterations = max_iterations
        
        self._langchain_tools = [self._convert_tool(t) for t in tools]
        
        self._agent_graph = create_agent(
            model=self._llm,
            tools=self._langchain_tools,
            system_prompt=REACT_SYSTEM_PROMPT,
            debug=False,
        )
        
        logger.info(f"ReActAgent initialized with {len(tools)} tools: {[t.name for t in tools]}")
    
    def _convert_tool(self, tool: BaseTool) -> StructuredTool:
        """Convert custom BaseTool to LangChain StructuredTool."""
        from pydantic import BaseModel, Field
        
        class ToolInput(BaseModel):
            query: str = Field(description="Input query for the tool")
        
        def tool_func(query: str) -> str:
            context = tool.get_context()
            result: ToolResult = tool.execute(query, context)
            logger.info(f"[ReAct] Tool {tool.name} executed, success={result.success}, output_len={len(result.output)}")
            return result.output
        
        return StructuredTool(
            name=tool.name,
            description=tool.description,
            func=tool_func,
            args_schema=ToolInput,
        )
    
    def run(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent with tool calling loop."""
        try:
            inputs = {"messages": [{"role": "user", "content": query}]}
            
            all_tool_calls = []
            final_output = ""
            
            result = self._agent_graph.invoke(inputs)
            logger.info(f"[ReAct] Invoke result keys: {list(result.keys())}")
            
            messages = result.get("messages", [])
            logger.info(f"[ReAct] Total messages: {len(messages)}")
            
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                logger.info(f"[ReAct] Msg {i}: type={msg_type}")
                
                if isinstance(msg, AIMessage):
                    tc_list = msg.tool_calls or []
                    logger.info(f"[ReAct] AIMessage tool_calls: {len(tc_list)}")
                    for tc in tc_list:
                        if isinstance(tc, dict):
                            name = tc.get("name", "unknown")
                        else:
                            name = getattr(tc, "name", str(tc))
                        all_tool_calls.append(name)
                        logger.info(f"[ReAct] Tool call recorded: {name}")
                    
                    if msg.content and not tc_list:
                        content_raw = msg.content
                        content_type = type(content_raw).__name__
                        logger.info(f"[ReAct] AIMessage content type: {content_type}, raw preview: {str(content_raw)[:200]}")
                        content = self._extract_text_content(content_raw)
                        if content:
                            final_output = content
                            logger.info(f"[ReAct] Final content extracted: {len(content)} chars, preview: {content[:100]}")
                
                elif isinstance(msg, ToolMessage):
                    logger.info(f"[ReAct] ToolMessage: {msg.tool_call_id}")
            
            if not final_output and all_tool_calls:
                final_output = self._fallback_summary(query, all_tool_calls)
                logger.info(f"[ReAct] Fallback summary generated")
            
            success = bool(final_output)
            output_str = self._extract_text_content(final_output) if final_output else ""
            logger.info(f"[ReAct] Final: tool_calls={all_tool_calls}, output_len={len(output_str)}, success={success}")
            
            return {
                "success": success,
                "output": output_str or self._fallback_summary(query, all_tool_calls) or "抱歉，处理请求时未能获得有效结果。",
                "tool_calls": all_tool_calls,
            }
            
        except Exception as e:
            logger.exception(f"ReActAgent execution failed: {e}")
            return {
                "success": False,
                "output": f"抱歉，处理您的请求时出现错误: {str(e)[:100]}",
                "tool_calls": [],
            }
    
    def _fallback_summary(self, query: str, tool_calls: List[str]) -> str:
        """Generate fallback summary when agent doesn't produce final content."""
        if "rag" in tool_calls:
            return f"根据知识库检索，已查询与「{query}」相关的内容。请查看上述检索结果。"
        elif "web_search" in tool_calls:
            return f"通过网络搜索，已获取与「{query}」相关的信息。请查看上述搜索结果。"
        else:
            return f"已处理您的查询「{query}」，请查看处理结果。"
    
    def _extract_text_content(self, content: Any) -> str:
        """Extract text from content that may be string or list of blocks."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            return "".join(text_parts) if text_parts else ""
        else:
            return str(content) if content else ""
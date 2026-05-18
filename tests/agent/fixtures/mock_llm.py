import json
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage


class MockLLMWrapper:
    def __init__(
        self,
        responses: List[AIMessage] = None,
        tool_call_sequence: List[List[Dict]] = None,
        final_answer: str = None,
    ):
        self.responses = responses or []
        self.tool_call_sequence = tool_call_sequence or []
        self.final_answer = final_answer or "这是最终答案"
        self._call_count = 0
        self._invoke_count = 0
    
    def invoke_with_tools(self, messages: List, tools: List) -> AIMessage:
        self._call_count += 1
        
        if self.responses and self._call_count <= len(self.responses):
            return self.responses[self._call_count - 1]
        
        if self.tool_call_sequence and self._call_count <= len(self.tool_call_sequence):
            tool_calls = self.tool_call_sequence[self._call_count - 1]
            return AIMessage(content="", tool_calls=tool_calls)
        
        return AIMessage(content=self.final_answer)
    
    def invoke(self, user_content: str, system: str = None) -> tuple:
        self._invoke_count += 1
        return (self.final_answer, 100, 50)
    
    def get_call_count(self) -> int:
        return self._call_count


def create_mock_llm_with_tool_calls(tool_calls: List[Dict], final_answer: str = None) -> MagicMock:
    mock_llm = MagicMock()
    call_count = 0
    final = final_answer or "根据检索结果，答案如下..."
    
    def invoke_with_tools_impl(messages, tools):
        nonlocal call_count
        call_count += 1
        if call_count <= len(tool_calls):
            return AIMessage(content="", tool_calls=[tool_calls[call_count - 1]])
        return AIMessage(content=final)
    
    mock_llm.invoke_with_tools = invoke_with_tools_impl
    return mock_llm


def create_mock_llm_with_answer(answer: str) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.invoke_with_tools = lambda messages, tools: AIMessage(content=answer)
    return mock_llm


def create_mock_llm_sequence(responses: List[AIMessage]) -> MagicMock:
    mock_llm = MagicMock()
    call_count = 0
    
    def invoke_with_tools_impl(messages, tools):
        nonlocal call_count
        call_count += 1
        if call_count <= len(responses):
            return responses[call_count - 1]
        return AIMessage(content="默认答案")
    
    mock_llm.invoke_with_tools = invoke_with_tools_impl
    return mock_llm
"""
LangChain LLM Component for MiniMax API

Creates ChatAnthropic instance configured for MiniMax endpoint.
"""
import os
import threading
from typing import Optional

from langchain_anthropic import ChatAnthropic


def create_minimax_llm(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    timeout_seconds: int = 120,
) -> ChatAnthropic:
    """
    Create ChatAnthropic instance for MiniMax API.

    Args:
        api_key: MiniMax API key (defaults to LLM_API_KEY env var)
        endpoint: MiniMax Anthropic endpoint (defaults to LLM_ENDPOINT env var)
        model: Model name (defaults to LLM_MODEL env var or MiniMax-M2.7)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        timeout_seconds: Request timeout

    Returns:
        ChatAnthropic instance configured for MiniMax
    """
    api_key = api_key or os.getenv("LLM_API_KEY", "")
    endpoint = endpoint or os.getenv("LLM_ENDPOINT", "https://api.minimaxi.com/anthropic")
    model = model or os.getenv("LLM_MODEL", "MiniMax-M2.7")

    if not api_key:
        raise ValueError("LLM_API_KEY not provided or not in environment")

    return ChatAnthropic(
        model=model,
        api_key=api_key,
        anthropic_api_url=endpoint,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout_seconds,
    )


class MiniMaxLLMWrapper:
    """
    Wrapper for MiniMax LLM with thinking parameter support.

    MiniMax API supports extended thinking (like Claude), which can
    significantly increase latency. This wrapper allows disabling
    thinking for faster responses.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        disable_thinking: bool = True,
        timeout_seconds: int = 120,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.endpoint = endpoint or os.getenv("LLM_ENDPOINT", "")
        self.model = model or os.getenv("LLM_MODEL", "MiniMax-M2.7")
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        self.timeout_seconds = timeout_seconds
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self) -> ChatAnthropic:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = create_minimax_llm(
                        api_key=self.api_key,
                        endpoint=self.endpoint,
                        model=self.model,
                        max_tokens=self.max_tokens,
                        timeout_seconds=self.timeout_seconds,
                    )
        return self._client

    def invoke(self, prompt: str, system: Optional[str] = None) -> tuple[str, int, int]:
        """
        Invoke LLM and return text content with token counts.

        Args:
            prompt: User input prompt
            system: System prompt (optional)

        Returns:
            Tuple of (generated text content, input_tokens, output_tokens)
        """
        client = self._get_client()

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [HumanMessage(content=prompt)]
        if system:
            messages = [SystemMessage(content=system)] + messages

        response = client.invoke(messages)

        token_usage = response.response_metadata.get("token_usage", {})
        input_tokens = token_usage.get("prompt_tokens", 0)
        output_tokens = token_usage.get("completion_tokens", 0)

        text_content = ""
        for block in response.content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_content += block.get("text", "")
            elif hasattr(block, "type"):
                if block.type == "text":
                    text_content += block.text
            elif isinstance(block, str):
                text_content += block

        return text_content.strip(), input_tokens, output_tokens
    
    def invoke_text(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Invoke LLM and return only text content (backward compatible).

        Args:
            prompt: User input prompt
            system: System prompt (optional)

        Returns:
            Generated text content
        """
        text, _, _ = self.invoke(prompt, system)
        return text
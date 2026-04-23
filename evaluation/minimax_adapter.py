"""
MiniMax LLM Adapter for Ragas Evaluation

Converts Anthropic API format to Ragas-compatible interface.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import anthropic


class MiniMaxRagasAdapter:
    """
    Adapter for MiniMax API via Anthropic SDK.
    
    Usage with Ragas:
    ```python
    from evaluation.minimax_adapter import MiniMaxRagasAdapter
    from ragas import evaluate
    from ragas.metrics import faithfulness
    
    adapter = MiniMaxRagasAdapter()
    # Use adapter's generate method for Ragas evaluation
    ```
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: str = "MiniMax-M2.7",
        timeout_seconds: int = 30,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.endpoint = endpoint or os.getenv("LLM_ENDPOINT", "")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._client = None
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY not provided or not in environment")
        
        if not self.endpoint:
            raise ValueError("LLM_ENDPOINT not provided or not in environment")
    
    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.endpoint,
            )
        return self._client
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate response using MiniMax API via Anthropic SDK.
        
        Args:
            prompt: User input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system: System prompt (optional)
            **kwargs: Additional parameters
        
        Returns:
            Generated text content
        """
        client = self._get_client()
        
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
            
            # Extract only text blocks (ignore thinking blocks)
            text_content = ""
            for block in message.content:
                if block.type == "text":
                    text_content += block.text
            
            return text_content.strip()
            
        except anthropic.APIError as e:
            raise RuntimeError(f"MiniMax API error: {e.__class__.__name__}: {str(e)[:200]}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e.__class__.__name__}: {str(e)[:200]}")
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 2048,
        temperature: float = 0.1,
        system: Optional[str] = None,
    ) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of user input prompts
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            system: System prompt (optional)
        
        Returns:
            List of generated text content
        """
        results = []
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
            )
            results.append(result)
        return results
    
    def is_available(self) -> bool:
        """Check if adapter is properly configured"""
        return bool(self.api_key) and bool(self.endpoint)
    
    def get_model_name(self) -> str:
        """Get current model name"""
        return self.model


def create_minimax_adapter_from_env() -> MiniMaxRagasAdapter:
    """
    Create MiniMaxRagasAdapter from environment variables.
    
    Required env vars:
    - LLM_API_KEY
    - LLM_ENDPOINT
    - LLM_MODEL (optional, defaults to MiniMax-M2.7)
    """
    return MiniMaxRagasAdapter(
        api_key=os.getenv("LLM_API_KEY"),
        endpoint=os.getenv("LLM_ENDPOINT"),
        model=os.getenv("LLM_MODEL", "MiniMax-M2.7"),
    )
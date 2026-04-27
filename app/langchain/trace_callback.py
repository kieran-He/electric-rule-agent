"""
LangChain Callback Handler for automatic trace recording.

Records:
- LLM start/end time, Token usage
- Retrieval start/end time, document count
- Chain execution steps
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from langchain.callbacks.base import BaseCallbackHandler


class TraceCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for automatic trace recording.
    
    Records:
    - LLM start/end time, Token usage
    - Retrieval start/end time, document count
    """

    def __init__(self, trace_id: str, session_id: str):
        self.trace_id = trace_id
        self.session_id = session_id
        self._llm_start_time: float | None = None
        self._retrieval_start_time: float | None = None
        self._token_usage: Dict[str, int] = {}
        self._retrieval_stats: Dict[str, Any] = {}
        self._errors: List[Dict[str, Any]] = []

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        self._llm_start_time = time.time()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        if self._llm_start_time is None:
            return
        
        llm_latency = int((time.time() - self._llm_start_time) * 1000)
        
        token_usage = {}
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
        
        self._token_usage = {
            "input_tokens": token_usage.get("prompt_tokens", 0),
            "output_tokens": token_usage.get("completion_tokens", 0),
            "llm_latency_ms": llm_latency,
        }

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        self._errors.append({
            "type": "llm_error",
            "message": str(error),
            "timestamp": time.time(),
        })

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        **kwargs: Any,
    ) -> None:
        self._retrieval_start_time = time.time()

    def on_retriever_end(self, documents: List[Any], **kwargs: Any) -> None:
        if self._retrieval_start_time is None:
            return
        
        retrieval_latency = int((time.time() - self._retrieval_start_time) * 1000)
        self._retrieval_stats = {
            "doc_count": len(documents),
            "retrieval_latency_ms": retrieval_latency,
        }

    def on_retriever_error(self, error: Exception, **kwargs: Any) -> None:
        self._errors.append({
            "type": "retriever_error",
            "message": str(error),
            "timestamp": time.time(),
        })

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        self._errors.append({
            "type": "chain_error",
            "message": str(error),
            "timestamp": time.time(),
        })

    def get_token_usage(self) -> Dict[str, int]:
        return self._token_usage

    def get_retrieval_stats(self) -> Dict[str, Any]:
        return self._retrieval_stats

    def get_errors(self) -> List[Dict[str, Any]]:
        return self._errors
    
    def has_errors(self) -> bool:
        return len(self._errors) > 0
    
    def get_first_error(self) -> Dict[str, Any] | None:
        return self._errors[0] if self._errors else None
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "token_usage": self._token_usage,
            "retrieval_stats": self._retrieval_stats,
            "errors": self._errors,
            "success": not self.has_errors(),
        }
from typing import Dict, List, Optional

import anthropic

from app.repository import PolicyChunk


class LLMGenerationError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)


class LLMClient:
    def __init__(self, api_key: str, endpoint: str, model: str, timeout_seconds: int = 30, provider: str = "anthropic"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.provider = provider
        self._client = None

    @property
    def ready(self) -> bool:
        return bool(self.api_key)

    @property
    def mode(self) -> str:
        return "llm" if self.ready else "unavailable"

    def _require_ready(self) -> None:
        if not self.ready:
            raise LLMGenerationError("llm unavailable: API_KEY is empty")

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.endpoint,
            )
        return self._client

    def _build_context(self, chunks: List[PolicyChunk], title: str) -> str:
        lines = [title]
        if not chunks:
            lines.append("- none")
            return "\n".join(lines)
        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source_name", "unknown")
            snippet = chunk.text[:260]
            lines.append(f"{idx}. [{source}] {snippet}")
        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        return (
            "你是电力政策问答助手。只能根据提供的证据回答，禁止编造。"
            "如果证据不足，明确说明\"未检索到充分依据\"。"
        )

    def _call_anthropic_sdk(self, system: str, user_content: str) -> str:
        self._require_ready()
        client = self._get_client()
        
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            
            text_content = ""
            for block in message.content:
                if block.type == "text":
                    text_content += block.text
            
            text_content = text_content.strip()
            if not text_content:
                raise LLMGenerationError("empty text content from upstream")
            
            return text_content
            
        except anthropic.APIError as e:
            raise LLMGenerationError(f"anthropic api error: {e.__class__.__name__}: {str(e)[:200]}")
        except Exception as e:
            if isinstance(e, LLMGenerationError):
                raise
            raise LLMGenerationError(f"unexpected error: {e.__class__.__name__}: {str(e)[:200]}")

    def generate_answer(
        self,
        query: str,
        provincial_chunks: List[PolicyChunk],
        global_chunks: List[PolicyChunk],
        history: List[str],
        province_code: Optional[str],
    ) -> str:
        self._require_ready()
        
        context = "\n\n".join(
            [
                self._build_context(provincial_chunks, f"省级证据({province_code or 'unknown'})"),
                self._build_context(global_chunks, "通用证据"),
                "历史对话:\n" + "\n".join(history[-3:]),
            ]
        )
        
        user_content = f"问题: {query}\n\n证据:\n{context}"
        return self._call_anthropic_sdk(self._build_system_prompt(), user_content)

    def generate_compare_answer(self, query: str, result_by_province: dict) -> str:
        self._require_ready()
        
        lines = [f"问题: {query}", "跨省检索证据:"]
        for province, chunks in result_by_province.items():
            lines.append(self._build_context(chunks, f"{province}证据"))
        
        user_content = "\n\n".join(lines)
        system = (
            "你是电力政策问答助手。请基于给定的跨省证据输出结论与差异点。"
            "没有证据时必须明确说明\"未检索到充分依据\"。"
        )
        return self._call_anthropic_sdk(system, user_content)


GLMClient = LLMClient
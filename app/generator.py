from typing import Dict, List, Optional

import requests

from app.repository import PolicyChunk


class LLMGenerationError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)


class GLMClient:
    def __init__(self, api_key: str, endpoint: str, model: str, timeout_seconds: int = 60):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds

    @property
    def ready(self) -> bool:
        return bool(self.api_key)

    @property
    def mode(self) -> str:
        return "llm" if self.ready else "unavailable"

    def _require_ready(self) -> None:
        if not self.ready:
            raise LLMGenerationError("llm unavailable: GLM_API_KEY is empty")

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

    def _build_payload(
        self,
        query: str,
        provincial_chunks: List[PolicyChunk],
        global_chunks: List[PolicyChunk],
        history: List[str],
        province_code: Optional[str],
    ) -> Dict[str, object]:
        prompt = (
            "你是电力政策问答助手。只能根据提供的证据回答，禁止编造。"
            "如果证据不足，明确说明“未检索到充分依据”。"
        )
        context = "\n\n".join(
            [
                self._build_context(provincial_chunks, f"省级证据({province_code or 'unknown'})"),
                self._build_context(global_chunks, "通用证据"),
                "历史对话:\n" + "\n".join(history[-3:]),
            ]
        )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"问题: {query}\n\n证据:\n{context}"},
            ],
            "temperature": 0.1,
        }

    def _build_compare_payload(self, query: str, result_by_province: dict) -> Dict[str, object]:
        lines = [f"问题: {query}", "跨省检索证据:"]
        for province, chunks in result_by_province.items():
            lines.append(self._build_context(chunks, f"{province}证据"))
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是电力政策问答助手。请基于给定的跨省证据输出结论与差异点。"
                        "没有证据时必须明确说明“未检索到充分依据”。"
                    ),
                },
                {"role": "user", "content": "\n\n".join(lines)},
            ],
            "temperature": 0.1,
        }

    def _call_llm(self, payload: Dict[str, object]) -> str:
        self._require_ready()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err: Exception | None = None
        for _ in range(2):
            try:
                with requests.Session() as session:
                    # Avoid inheriting broken global proxy settings from host env.
                    session.trust_env = False
                    response = session.post(
                        self.endpoint, json=payload, headers=headers, timeout=self.timeout_seconds
                    )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise LLMGenerationError("empty model choices from upstream")
                message = choices[0].get("message") or {}
                content = str(message.get("content", "")).strip()
                if not content:
                    raise LLMGenerationError("empty model content from upstream")
                return content
            except requests.ReadTimeout as exc:
                last_err = exc
                continue
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                raise LLMGenerationError(f"upstream http error: status={status}") from exc
            except requests.RequestException as exc:
                raise LLMGenerationError(f"upstream request failed: {exc.__class__.__name__}") from exc
            except (TypeError, ValueError, KeyError) as exc:
                if isinstance(exc, LLMGenerationError):
                    raise
                raise LLMGenerationError("invalid model response payload") from exc
        if last_err is not None:
            raise LLMGenerationError("upstream timeout")
        raise LLMGenerationError("model call failed without explicit exception")

    def generate_answer(
        self,
        query: str,
        provincial_chunks: List[PolicyChunk],
        global_chunks: List[PolicyChunk],
        history: List[str],
        province_code: Optional[str],
    ) -> str:
        self._require_ready()
        payload = self._build_payload(query, provincial_chunks, global_chunks, history, province_code)
        return self._call_llm(payload)

    def generate_compare_answer(self, query: str, result_by_province: dict) -> str:
        self._require_ready()
        payload = self._build_compare_payload(query, result_by_province)
        return self._call_llm(payload)

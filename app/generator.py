from typing import Dict, List, Optional

import requests

from app.repository import PolicyChunk


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
        return "llm" if self.ready else "fallback"

    def _build_context(self, chunks: List[PolicyChunk], title: str) -> str:
        lines = [title]
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
            "输出要简洁并说明是否存在省级与通用规则差异。"
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

    def _call_llm(self, payload: Dict[str, object]) -> str:
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
                    raise ValueError("empty choices returned by model api")
                message = choices[0].get("message") or {}
                content = str(message.get("content", "")).strip()
                if not content:
                    raise ValueError("empty content returned by model api")
                return content
            except requests.ReadTimeout as exc:
                last_err = exc
                continue
        if last_err is not None:
            raise last_err
        raise ValueError("model call failed without explicit exception")

    def generate_answer(
        self,
        query: str,
        provincial_chunks: List[PolicyChunk],
        global_chunks: List[PolicyChunk],
        history: List[str],
        province_code: Optional[str],
    ) -> str:
        if not self.api_key:
            return self._fallback_answer(query, provincial_chunks, global_chunks, province_code)

        payload = self._build_payload(query, provincial_chunks, global_chunks, history, province_code)
        try:
            return self._call_llm(payload)
        except (requests.RequestException, ValueError, TypeError, KeyError):
            return self._fallback_answer(query, provincial_chunks, global_chunks, province_code)

    def generate_compare_answer(self, query: str, result_by_province: dict) -> str:
        lines = [f"问题: {query}", "跨省对比摘要:"]
        for province, chunks in result_by_province.items():
            if not chunks:
                lines.append(f"- {province}: 未检索到充分依据")
                continue
            lines.append(f"- {province}: {chunks[0].text[:140]}...")
        return "\n".join(lines)

    def _fallback_answer(
        self,
        query: str,
        provincial_chunks: List[PolicyChunk],
        global_chunks: List[PolicyChunk],
        province_code: Optional[str],
    ) -> str:
        if not provincial_chunks and not global_chunks:
            return "未检索到充分依据。请补充省份、交易类型或时间范围后重试。"
        summary_parts = []
        if provincial_chunks:
            summary_parts.append(f"{province_code or '省级'}依据: {provincial_chunks[0].text[:120]}")
        if global_chunks:
            summary_parts.append(f"通用依据: {global_chunks[0].text[:120]}")
        return f"基于检索结果，关于“{query}”的结论如下：\n" + "\n".join(summary_parts)

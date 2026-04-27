from types import SimpleNamespace

import pytest

from app.core.llm_client import GLMClient, LLMGenerationError


def _chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, metadata={"source_name": "demo.txt"})


def test_generate_answer_uses_llm_when_key_present(monkeypatch):
    client = GLMClient(api_key="test-key", endpoint="https://example.com", model="glm-test")

    monkeypatch.setattr(client, "_call_anthropic_sdk", lambda system, user: "模型回答")
    answer = client.generate_answer(
        query="陕西交易规则",
        provincial_chunks=[_chunk("省级证据A")],
        global_chunks=[_chunk("通用证据B")],
        history=[],
        province_code="SN",
    )
    assert answer == "模型回答"
    assert client.mode == "llm"


def test_generate_answer_raises_on_llm_failure(monkeypatch):
    client = GLMClient(api_key="test-key", endpoint="https://example.com", model="glm-test")

    def fake_call(system, user):
        raise LLMGenerationError("upstream timeout")

    monkeypatch.setattr(client, "_call_anthropic_sdk", fake_call)
    with pytest.raises(LLMGenerationError, match="upstream timeout"):
        client.generate_answer(
            query="陕西交易规则",
            provincial_chunks=[_chunk("省级证据A")],
            global_chunks=[],
            history=[],
            province_code="SN",
        )


def test_generate_answer_raises_without_key():
    client = GLMClient(api_key="", endpoint="https://example.com", model="glm-test")
    with pytest.raises(LLMGenerationError, match="API_KEY is empty"):
        client.generate_answer(
            query="陕西交易规则",
            provincial_chunks=[],
            global_chunks=[],
            history=[],
            province_code=None,
        )
    assert client.mode == "unavailable"


def test_generate_compare_answer_uses_llm(monkeypatch):
    client = GLMClient(api_key="test-key", endpoint="https://example.com", model="glm-test")
    monkeypatch.setattr(client, "_call_anthropic_sdk", lambda system, user: "跨省比较结论")

    out = client.generate_compare_answer(
        query="比较陕西和广东中长期交易规则",
        result_by_province={"SN": [_chunk("证据A")], "GD": [_chunk("证据B")]},
    )

    assert out == "跨省比较结论"
from types import SimpleNamespace

import requests

from app.generator import GLMClient


def _chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, metadata={"source_name": "demo.txt"})


def test_generate_answer_uses_llm_when_key_present(monkeypatch):
    client = GLMClient(api_key="test-key", endpoint="https://example.com", model="glm-test")

    monkeypatch.setattr(client, "_call_llm", lambda payload: "模型回答")
    answer = client.generate_answer(
        query="陕西交易规则",
        provincial_chunks=[_chunk("省级证据A")],
        global_chunks=[_chunk("通用证据B")],
        history=[],
        province_code="SN",
    )
    assert answer == "模型回答"
    assert client.mode == "llm"


def test_generate_answer_falls_back_on_llm_failure(monkeypatch):
    client = GLMClient(api_key="test-key", endpoint="https://example.com", model="glm-test")

    def fake_call(_payload):
        raise requests.RequestException("network failed")

    monkeypatch.setattr(client, "_call_llm", fake_call)
    answer = client.generate_answer(
        query="陕西交易规则",
        provincial_chunks=[_chunk("省级证据A")],
        global_chunks=[],
        history=[],
        province_code="SN",
    )
    assert "基于检索结果" in answer


def test_generate_answer_fallback_without_key():
    client = GLMClient(api_key="", endpoint="https://example.com", model="glm-test")
    answer = client.generate_answer(
        query="陕西交易规则",
        provincial_chunks=[],
        global_chunks=[],
        history=[],
        province_code=None,
    )
    assert "未检索到充分依据" in answer
    assert client.mode == "fallback"

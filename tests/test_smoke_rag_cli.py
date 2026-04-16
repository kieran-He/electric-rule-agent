from types import SimpleNamespace

import pytest

from app.generator import LLMGenerationError
from app.schemas import QueryMode
from tools import smoke_rag


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def test_main_default_args_success(monkeypatch, capsys):
    monkeypatch.setattr(
        smoke_rag,
        'health',
        lambda: DummyResponse({'status': 'ok', 'vector_store_ready': True, 'glm_ready': True, 'message': 'ok'}),
    )

    captured_req = {}

    def fake_process(req):
        captured_req['req'] = req
        return DummyResponse({'mode': QueryMode.province_plus_global.value, 'province_code': 'SN'})

    monkeypatch.setattr(smoke_rag.service, 'process', fake_process)

    code = smoke_rag.main([])
    out = capsys.readouterr().out

    assert code == 0
    assert '[health]' in out
    assert '[query]' in out
    assert captured_req['req'].province_codes == ['SN']
    assert captured_req['req'].mode == QueryMode.auto


def test_main_accepts_cli_args(monkeypatch):
    monkeypatch.setattr(smoke_rag, 'health', lambda: DummyResponse({'status': 'ok'}))

    captured_req = {}

    def fake_process(req):
        captured_req['req'] = req
        return DummyResponse({'mode': QueryMode.single_province.value})

    monkeypatch.setattr(smoke_rag.service, 'process', fake_process)

    code = smoke_rag.main(
        [
            '--query',
            '陕西中长期交易流程是什么',
            '--province',
            'gd',
            '--top-k',
            '7',
            '--session-id',
            'abc',
            '--mode',
            'single_province',
        ]
    )

    assert code == 0
    req = captured_req['req']
    assert req.query == '陕西中长期交易流程是什么'
    assert req.province_codes == ['GD']
    assert req.top_k == 7
    assert req.session_id == 'abc'
    assert req.mode == QueryMode.single_province


def test_main_returns_nonzero_on_query_failure(monkeypatch, capsys):
    monkeypatch.setattr(smoke_rag, 'health', lambda: DummyResponse({'status': 'ok'}))

    def fake_process(_req):
        raise LLMGenerationError('upstream timeout')

    monkeypatch.setattr(smoke_rag.service, 'process', fake_process)

    code = smoke_rag.main(['--query', '陕西中长期交易流程'])
    out = capsys.readouterr().out

    assert code == 1
    assert 'upstream timeout' in out


def test_invalid_top_k_exits_with_2(monkeypatch):
    monkeypatch.setattr(smoke_rag, 'health', lambda: DummyResponse({'status': 'ok'}))
    monkeypatch.setattr(smoke_rag.service, 'process', lambda _req: DummyResponse({'mode': 'x'}))

    with pytest.raises(SystemExit) as exc:
        smoke_rag.main(['--top-k', '0'])

    assert exc.value.code == 2

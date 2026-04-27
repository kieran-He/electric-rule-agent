from unittest.mock import patch, MagicMock
import pytest

from tools import smoke_rag


class FakeResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data
    
    def json(self):
        return self._json_data


def test_main_default_args_success(monkeypatch, capsys):
    fake_client = MagicMock()
    fake_client.get.return_value = FakeResponse(200, {'overall': 'ok', 'components': []})
    fake_client.post.return_value = FakeResponse(200, {'answer': 'test answer', 'intent': 'clause_qa'})
    
    monkeypatch.setattr(smoke_rag, 'TestClient', lambda app: fake_client)
    
    code = smoke_rag.main([])
    out = capsys.readouterr().out
    
    assert code == 0
    assert '[health]' in out
    assert '[query]' in out


def test_main_accepts_cli_args(monkeypatch):
    fake_client = MagicMock()
    fake_client.get.return_value = FakeResponse(200, {'overall': 'ok'})
    fake_client.post.return_value = FakeResponse(200, {'answer': 'test answer'})
    
    monkeypatch.setattr(smoke_rag, 'TestClient', lambda app: fake_client)
    
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
        ]
    )
    
    assert code == 0
    call_args = fake_client.post.call_args
    assert call_args[1]['json']['query'] == '陕西中长期交易流程是什么'
    assert call_args[1]['json']['province_codes'] == ['GD']
    assert call_args[1]['json']['top_k'] == 7
    assert call_args[1]['json']['session_id'] == 'abc'


def test_main_returns_nonzero_on_query_failure(monkeypatch, capsys):
    fake_client = MagicMock()
    fake_client.get.return_value = FakeResponse(200, {'overall': 'ok'})
    fake_client.post.return_value = FakeResponse(503, {'detail': 'upstream timeout'})
    
    monkeypatch.setattr(smoke_rag, 'TestClient', lambda app: fake_client)
    
    code = smoke_rag.main(['--query', '陕西中长期交易流程'])
    
    assert code == 1


def test_invalid_top_k_exits_with_2(monkeypatch):
    fake_client = MagicMock()
    fake_client.get.return_value = FakeResponse(200, {'overall': 'ok'})
    fake_client.post.return_value = FakeResponse(200, {'answer': 'ok'})
    
    monkeypatch.setattr(smoke_rag, 'TestClient', lambda app: fake_client)
    
    with pytest.raises(SystemExit) as exc:
        smoke_rag.main(['--top-k', '0'])
    
    assert exc.value.code == 2
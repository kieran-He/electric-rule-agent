import pytest
from pydantic import ValidationError

from app.schemas import QueryRequest


def test_query_request_accepts_valid_top_k():
    req = QueryRequest(query="test", session_id="s1", top_k=5)
    assert req.top_k == 5


def test_query_request_rejects_top_k_below_one():
    with pytest.raises(ValidationError):
        QueryRequest(query="test", session_id="s1", top_k=0)


def test_query_request_rejects_top_k_too_large():
    with pytest.raises(ValidationError):
        QueryRequest(query="test", session_id="s1", top_k=21)

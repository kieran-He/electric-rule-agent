import pytest
from pydantic import ValidationError

from app.schemas.query import QueryRequest


def test_query_request_accepts_valid_top_k():
    req = QueryRequest(query="test", session_id="s1", top_k=5)
    assert req.top_k == 5


def test_query_request_rejects_top_k_below_one():
    with pytest.raises(ValidationError):
        QueryRequest(query="test", session_id="s1", top_k=0)


def test_query_request_rejects_top_k_too_large():
    with pytest.raises(ValidationError):
        QueryRequest(query="test", session_id="s1", top_k=21)


def test_query_request_default_province_codes():
    req = QueryRequest(query="test", session_id="s1")
    assert req.province_codes == ["SN"]


def test_query_request_custom_province_codes():
    req = QueryRequest(query="test", session_id="s1", province_codes=["GD", "SN"])
    assert req.province_codes == ["GD", "SN"]
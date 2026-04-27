from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.query import QueryRequest
from app.schemas.answer import QueryAnswer
from app.services.query_service import QueryService


class FakeSession:
    def __init__(self):
        self._committed = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    
    def commit(self):
        self._committed = True
    
    def scalars(self, stmt):
        return MagicMock(all=lambda: [])
    
    def add(self, obj):
        pass
    
    def flush(self):
        pass


class FakeOrchestrator:
    def __init__(self, db, settings):
        self.db = db
        self.settings = settings
    
    def run(self, req, history=None, trace_service=None):
        return QueryAnswer(
            answer="测试结论",
            citations=[],
            intent="clause_qa",
            confidence=0.8,
            used_documents=["陕西规则.pdf"],
            trace_id="trace_test",
            flow=None,
            warnings=[],
        )


def build_service():
    settings = SimpleNamespace(
        chroma_path="./data/chroma",
        embedding_model="deterministic",
        hybrid_vector_top_k=8,
        hybrid_bm25_top_k=8,
        hybrid_final_top_k=8,
        reranker_model="BAAI/bge-reranker-large",
        reranker_preload=False,
        reranker_max_length=512,
        bm25_k1=1.5,
        bm25_b=0.6,
        query_expansion=False,
        query_expansion_method="synonyms",
        query_expansion_max=3,
        query_rewrite_enabled=False,
        query_rewrite_min_length=10,
        query_rewrite_keep_original=True,
    )
    
    session_factory = lambda: FakeSession()
    
    return QueryService(settings=settings, session_factory=session_factory)


def test_query_service_returns_answer():
    service = build_service()
    
    with patch('app.services.query_service.HybridQAOrchestrator', FakeOrchestrator):
        with patch.object(service.conversation_service, 'get_history', return_value=[]):
            with patch.object(service.conversation_service, 'append_turn'):
                req = QueryRequest(
                    query="陕西的交易流程是什么？",
                    session_id="s1",
                    province_codes=["SN"],
                )
                resp = service.answer(req)
                
                assert resp.answer == "测试结论"
                assert resp.intent == "clause_qa"
                assert resp.used_documents == ["陕西规则.pdf"]
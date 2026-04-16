from app.province import ProvinceDetector
from app.schemas import QueryMode, QueryRequest
from app.service import PolicyQueryService, QueryPlanner
from app.session import SessionStore


class FakeChunk:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata
        self.score = 0.9


class FakeRepo:
    def retrieve(self, query, top_k, kb_scope, province_code):
        if kb_scope == "province" and province_code == "SN":
            return [
                FakeChunk(
                    "陕西中长期交易按月组织开展。",
                    {
                        "province_code": "SN",
                        "source_name": "陕西规则.pdf",
                        "doc_id": "sn-1",
                        "policy_level": "province",
                        "effective_date": "2026-01-01",
                    },
                )
            ]
        if kb_scope == "global":
            return [
                FakeChunk(
                    "全国规则要求交易公开透明。",
                    {
                        "province_code": "",
                        "source_name": "全国规则.pdf",
                        "doc_id": "cn-1",
                        "policy_level": "national",
                        "effective_date": "2025-01-01",
                    },
                )
            ]
        return []


class FakeGenerator:
    def generate_answer(self, query, provincial_chunks, global_chunks, history, province_code):
        return "测试结论"

    def generate_compare_answer(self, query, result_by_province):
        return "对比结论"


def build_service():
    return PolicyQueryService(
        repository=FakeRepo(),
        generator=FakeGenerator(),
        detector=ProvinceDetector(),
        sessions=SessionStore(),
        planner=QueryPlanner(),
    )


def test_auto_mode_resolves_to_province_plus_global():
    service = build_service()
    req = QueryRequest(query="陕西的交易流程是什么？", session_id="s1", mode=QueryMode.auto)
    resp = service.process(req)
    assert resp.mode == QueryMode.province_plus_global
    assert resp.province_code == "SN"
    assert len(resp.provincial_evidence) == 1
    assert len(resp.global_evidence) == 1


def test_need_province_confirmation():
    service = build_service()
    req = QueryRequest(query="交易流程是什么？", session_id="s2", mode=QueryMode.single_province)
    resp = service.process(req)
    assert resp.needs_confirmation is True
    assert resp.confirmation_question


def test_compare_mode_requires_two_provinces():
    service = build_service()
    req = QueryRequest(query="比较陕西交易规则", session_id="s3", mode=QueryMode.multi_province_compare)
    resp = service.process(req)
    assert resp.needs_confirmation is True

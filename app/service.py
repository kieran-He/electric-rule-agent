from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.generator import GLMClient
from app.province import PROVINCE_ALIASES, ProvinceDetector
from app.repository import ChromaPolicyRepository, PolicyChunk
from app.schemas import Citation, QueryMode, QueryRequest, QueryResponse
from app.session import SessionStore


def _to_citations(chunks: List[PolicyChunk], default_province: Optional[str] = None) -> List[Citation]:
    citations: List[Citation] = []
    for chunk in chunks:
        citations.append(
            Citation(
                province_code=chunk.metadata.get("province_code") or default_province,
                source_name=chunk.metadata.get("source_name", "unknown"),
                doc_id=chunk.metadata.get("doc_id", "unknown"),
                snippet=chunk.text[:200],
                policy_level=chunk.metadata.get("policy_level"),
                effective_date=chunk.metadata.get("effective_date"),
            )
        )
    return citations


class QueryPlanner:
    def resolve_mode(
        self, req: QueryRequest, detected_codes: List[str], active_province: Optional[str]
    ) -> QueryMode:
        if req.mode != QueryMode.auto:
            return req.mode
        if req.province_codes and len(req.province_codes) > 1:
            return QueryMode.multi_province_compare
        if len(detected_codes) > 1 or "对比" in req.query or "比较" in req.query:
            return QueryMode.multi_province_compare
        if req.province_codes or detected_codes or active_province:
            return QueryMode.province_plus_global
        return QueryMode.single_province

    def resolve_provinces(
        self,
        req: QueryRequest,
        detected_codes: List[str],
        active_province: Optional[str],
        mode: QueryMode,
    ) -> List[str]:
        if req.province_codes:
            return req.province_codes
        if mode == QueryMode.multi_province_compare:
            return detected_codes
        if detected_codes:
            return [detected_codes[0]]
        if active_province:
            return [active_province]
        return []


class PolicyQueryService:
    def __init__(
        self,
        repository: ChromaPolicyRepository,
        generator: GLMClient,
        detector: ProvinceDetector,
        sessions: SessionStore,
        planner: QueryPlanner,
    ):
        self.repository = repository
        self.generator = generator
        self.detector = detector
        self.sessions = sessions
        self.planner = planner

    def _detect_all_provinces(self, text: str) -> List[str]:
        codes: List[str] = []
        for alias, code in PROVINCE_ALIASES.items():
            if alias in text and code not in codes:
                codes.append(code)
        return codes

    def _retrieve_province_global(
        self, query: str, province_code: str, top_k: int
    ) -> Tuple[List[PolicyChunk], List[PolicyChunk]]:
        province_chunks = self.repository.retrieve(
            query=query, top_k=top_k, kb_scope="province", province_code=province_code
        )
        global_chunks = self.repository.retrieve(
            query=query, top_k=top_k, kb_scope="global", province_code=None
        )
        return province_chunks, global_chunks

    def _ask_llm_for_guidance(self, query: str, hint: str, history: List[str], province_code: Optional[str]) -> str:
        guided_query = f"{query}\n\n系统上下文：{hint}"
        return self.generator.generate_answer(
            query=guided_query,
            provincial_chunks=[],
            global_chunks=[],
            history=history,
            province_code=province_code,
        )

    def process(self, req: QueryRequest) -> QueryResponse:
        state = self.sessions.get(req.session_id)
        detector_output = self.detector.detect(req.query)
        detected_codes = self._detect_all_provinces(req.query)
        mode = self.planner.resolve_mode(req, detected_codes, state.active_province)
        provinces = self.planner.resolve_provinces(req, detected_codes, state.active_province, mode)
        top_k = req.top_k or settings.top_k

        if detector_output.province_code and detector_output.confidence >= settings.province_confidence_threshold:
            state.active_province = detector_output.province_code
            if detector_output.province_code not in provinces and mode != QueryMode.multi_province_compare:
                provinces = [detector_output.province_code]

        if mode in (QueryMode.single_province, QueryMode.province_plus_global) and not provinces:
            conclusion = self._ask_llm_for_guidance(
                req.query,
                "未识别到明确省份，请引导用户先确认省份后再回答具体规则。",
                state.history,
                None,
            )
            response = QueryResponse(
                mode=QueryMode.single_province,
                needs_confirmation=True,
                confirmation_question="请先确认省份，例如：陕西、广东、山东。",
                conclusion=conclusion,
                follow_up="请补充省份后重试，例如：陕西中长期交易流程。",
            )
            self.sessions.append_turn(req.session_id, req.query, conclusion)
            return response

        if mode == QueryMode.multi_province_compare:
            if len(provinces) < 2:
                conclusion = self._ask_llm_for_guidance(
                    req.query,
                    "当前跨省对比缺少足够省份，请引导用户至少提供两个省份。",
                    state.history,
                    None,
                )
                response = QueryResponse(
                    mode=mode,
                    needs_confirmation=True,
                    confirmation_question="请至少提供两个省份后再进行对比，例如：比较陕西和广东中长期交易规则。",
                    conclusion=conclusion,
                    follow_up="补充两个以上省份后，我会输出差异对比。",
                )
                self.sessions.append_turn(req.session_id, req.query, conclusion)
                return response

            result_by_province: Dict[str, List[PolicyChunk]] = {}
            all_citations: List[Citation] = []
            for province in provinces:
                chunks = self.repository.retrieve(
                    query=req.query, top_k=top_k, kb_scope="province", province_code=province
                )
                result_by_province[province] = chunks
                all_citations.extend(_to_citations(chunks, default_province=province))

            conclusion = self.generator.generate_compare_answer(req.query, result_by_province)
            response = QueryResponse(
                mode=mode,
                conclusion=conclusion,
                provincial_evidence=all_citations[: min(len(all_citations), 8)],
                differences="以上为跨省检索结果摘要，实际执行请以各省最新规则原文为准。",
                follow_up="可继续问：请按准入条件、价格机制、结算周期输出表格对比。",
            )
            self.sessions.append_turn(req.session_id, req.query, conclusion)
            return response

        province = provinces[0]
        provincial_chunks, global_chunks = self._retrieve_province_global(req.query, province, top_k)

        if not provincial_chunks and mode == QueryMode.single_province:
            answer = self._ask_llm_for_guidance(
                req.query,
                f"省份={province} 未检索到有效省级证据，请提示用户补充关键词。",
                state.history,
                province,
            )
            response = QueryResponse(
                mode=mode,
                province_code=province,
                conclusion=answer,
                follow_up="请补充更具体关键词，例如交易类型、年份、结算方式。",
            )
            self.sessions.append_turn(req.session_id, req.query, answer)
            return response

        answer = self.generator.generate_answer(
            query=req.query,
            provincial_chunks=provincial_chunks,
            global_chunks=global_chunks if mode == QueryMode.province_plus_global else [],
            history=state.history,
            province_code=province,
        )

        differences = None
        if mode == QueryMode.province_plus_global and provincial_chunks and global_chunks:
            differences = "如省级规则与通用规则冲突，已按省级口径优先解释。"

        response = QueryResponse(
            mode=mode,
            province_code=province,
            conclusion=answer,
            provincial_evidence=_to_citations(provincial_chunks, default_province=province),
            global_evidence=_to_citations(global_chunks) if mode == QueryMode.province_plus_global else [],
            differences=differences,
            follow_up="可继续问：请给出对应条款原文依据和执行注意事项。",
        )
        self.sessions.append_turn(req.session_id, req.query, answer)
        return response

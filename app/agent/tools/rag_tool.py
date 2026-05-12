from __future__ import annotations

import logging
from typing import Any, Dict, List, TYPE_CHECKING

from app.agent.tools.base import BaseTool, ToolResult
from app.schemas.answer import CitationItem
from app.schemas.query import QueryRequest

if TYPE_CHECKING:
    from app.langchain.orchestrator_hybrid import HybridQAOrchestrator

logger = logging.getLogger(__name__)


class RAGTool(BaseTool):
    name = "rag"
    description = "电力政策知识库检索工具。当用户询问电力政策、规则、交易等内容时使用。返回知识库相关文档或'知识库中未找到'提示。"
    keywords = [
        "电力", "电网", "售电", "发电", "用电", "电价",
        "交易", "市场", "政策", "规则", "规定", "通知",
        "条款", "准入", "结算", "偏差", "电量", "负荷",
        "现货", "中长期", "辅助服务", "容量", "调峰",
        "备用", "需求响应", "分布式", "新能源", "光伏",
        "风电", "储能", "输配电", "变电", "配电",
    ]
    
    def __init__(self, orchestrator: "HybridQAOrchestrator" = None):
        super().__init__()
        self._orchestrator = orchestrator
    
    def is_applicable(self, query: str) -> bool:
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.keywords)
    
    def execute(self, query: str, context: Dict[str, Any] = None) -> ToolResult:
        ctx = context or self._context
        
        orchestrator = ctx.get("orchestrator") or self._orchestrator
        if not orchestrator:
            return ToolResult(
                success=False,
                output="知识库服务暂不可用。",
                tool_name=self.name,
                confidence=0.0,
            )
        
        session_id = ctx.get("session_id", "")
        province_codes = ctx.get("province_codes", ["SN"])
        history = ctx.get("history", [])
        trace_service = ctx.get("trace_service", None)
        db = ctx.get("db", None)
        rewrite_result = ctx.get("rewrite_result", None)
        
        req = QueryRequest(
            query=query,
            session_id=session_id,
            province_codes=province_codes,
            top_k=ctx.get("top_k", 8),
            need_citation=ctx.get("need_citation", True),
        )
        
        try:
            answer = orchestrator.run(
                req,
                history=history,
                trace_service=trace_service,
                db=db,
                rewrite_result=rewrite_result,
            )
            
            citations: List[CitationItem] = []
            if answer.citations:
                citations = answer.citations
            
            no_result_indicators = ["未找到", "无相关信息", "无具体要求", "未检索到", "没有相关"]
            is_no_result = any(indicator in answer.answer for indicator in no_result_indicators)
            
            output = answer.answer
            
            missing = []
            if rewrite_result and rewrite_result.should_split:
                from dataprocess.province_mapping import PROVINCE_CODE_ALIASES
                
                provinces_status = []
                requested_codes = set()
                for qp in rewrite_result.queries:
                    requested_codes.update(qp.province_codes)
                
                province_names = [PROVINCE_CODE_ALIASES.get(code, code) for code in requested_codes]
                
                for province_name in province_names:
                    section_start = answer.answer.find(province_name)
                    logger.debug(f"[RAGTool] Province '{province_name}' at position {section_start}")
                    
                    if section_start == -1:
                        status = "未提及(需要补充)"
                    else:
                        next_province_pos = len(answer.answer)
                        for other_name in province_names:
                            other_pos = answer.answer.find(other_name)
                            if other_pos > section_start and other_pos < next_province_pos:
                                next_province_pos = other_pos
                        
                        section_text = answer.answer[section_start:next_province_pos]
                        logger.debug(f"[RAGTool] Section for '{province_name}' ({len(section_text)} chars): {section_text[:100]}...")
                        
                        has_no_result = any(ind in section_text for ind in no_result_indicators)
                        has_content = len(section_text) > 100 and not has_no_result
                        
                        logger.debug(f"[RAGTool] '{province_name}': has_no_result={has_no_result}, has_content={has_content}")
                        
                        if has_no_result:
                            status = "无结果(需要补充)"
                        elif has_content:
                            status = "有结果"
                        else:
                            status = "内容不足(需要补充)"
                    
                    provinces_status.append(f"{province_name}:{status}")
                
                status_prefix = f"【检索状态】{', '.join(provinces_status)}\n\n"
                output = status_prefix + answer.answer
                
                missing = [s.split(":")[0] for s in provinces_status if "需要补充" in s]
                if missing:
                    logger.info(f"[RAGTool] Provinces needing web_search: {missing}")
                    ctx["missing_provinces"] = missing
            
            return ToolResult(
                success=not is_no_result,
                output=output,
                metadata={
                    "intent": answer.intent,
                    "used_documents": answer.used_documents,
                    "warnings": answer.warnings,
                    "detected_provinces": answer.detected_provinces,
                    "is_no_result": is_no_result,
                    "missing_provinces": missing if rewrite_result and rewrite_result.should_split else [],
                },
                citations=citations,
                tool_name=self.name,
                confidence=answer.confidence,
            )
        except Exception as e:
            logger.exception(f"RAGTool execution failed: {e}")
            return ToolResult(
                success=False,
                output=f"知识库检索失败，请稍后重试。",
                tool_name=self.name,
                confidence=0.0,
            )
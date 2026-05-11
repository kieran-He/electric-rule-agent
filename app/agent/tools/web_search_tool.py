from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from app.agent.tools.base import BaseTool, ToolResult
from dataprocess.province_mapping import PROVINCE_CODE_ALIASES as PROVINCE_CODE_NAME

if TYPE_CHECKING:
    from app.core.web_search import WebSearchClient

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    name = "web_search"
    description = """网络搜索工具。用于补充知识库缺失的信息。

重要规则：
1. 只搜索用户明确提到的省份，不要搜索其他省份
2. 省份信息从系统提供的province_codes获取，如["SD", "SN"]表示山东、陕西
3. 搜索关键词格式："省份名 + 具体内容"，如"山东省中长期电力市场交易规则"
4. 如果province_codes包含多个省份，每个省份单独搜索一次"""
    keywords = ["最新", "新闻", "近期", "今天的", "最近", "最新消息", "实时", "当前"]
    
    def __init__(
        self,
        web_search_client: Optional["WebSearchClient"],
        llm_wrapper: Any,
        settings: Any,
    ):
        super().__init__()
        self._web_search_client = web_search_client
        self._llm_wrapper = llm_wrapper
        self._settings = settings
    
    def is_applicable(self, query: str) -> bool:
        return any(kw in query.lower() for kw in self.keywords)
    
    def execute(self, query: str, context: Dict[str, Any] = None) -> ToolResult:
        ctx = context or self._context
        
        if not self._web_search_client or not self._web_search_client.is_available():
            return ToolResult(
                success=False,
                output="网络搜索服务暂不可用。",
                tool_name=self.name,
                confidence=0.0,
            )
        
        province_codes = ctx.get("province_codes", [])
        
        if not province_codes:
            logger.warning("[WebSearch] No province_codes provided, skipping search")
            return ToolResult(
                success=False,
                output="未检测到省份信息，无法进行针对性搜索。",
                tool_name=self.name,
                confidence=0.0,
            )
        
        try:
            include_gov = getattr(self._settings, 'web_search_include_gov', True)
            domains = ["gov.cn"] if include_gov else None
            
            logger.info(f"[WebSearch] Searching for provinces: {province_codes}, query: {query}")
            
            return self._multi_province_search(query, province_codes, domains)
        
        except Exception as e:
            logger.exception(f"[WebSearch] Execution failed: {e}")
            return ToolResult(
                success=False,
                output="网络搜索失败，请稍后重试。",
                tool_name=self.name,
                confidence=0.0,
            )
    
    def _multi_province_search(
        self,
        query: str,
        province_codes: List[str],
        domains: Optional[List[str]],
    ) -> ToolResult:
        """Search for multiple provinces and combine results."""
        all_contexts: List[str] = []
        
        for code in province_codes:
            province_name = PROVINCE_CODE_NAME.get(code, code)
            province_query = f"{province_name} {query}"
            
            results = self._web_search_client.search(
                query=province_query,
                include_domains=domains,
            )
            
            if results:
                context = self._web_search_client.format_results_for_context(results)
                all_contexts.append(f"### {province_name} 搜索结果\n{context}")
        
        if not all_contexts:
            return ToolResult(
                success=False,
                output="网络搜索未找到相关信息。",
                tool_name=self.name,
                confidence=0.0,
            )
        
        combined_context = "\n\n".join(all_contexts)
        
        system_prompt = """你是搜索助手，根据网络搜索结果回答用户问题。涉及多省份时，分别说明各省份情况。

回答要求：
1. 基于搜索结果回答，不要编造信息
2. 禁止提及来源、证据出处等引用信息
3. 简洁清晰，直接回答问题核心
4. 涉及多省份时，分别说明各省份政策"""
        
        user_content = f"""问题：{query}

网络搜索结果：
{combined_context}

请根据上述搜索结果回答问题，分别说明各省份情况。"""
        
        answer, input_tokens, output_tokens = self._llm_wrapper.invoke(
            user_content, system=system_prompt
        )
        
        return ToolResult(
            success=True,
            output=f"⚠️ 此回答来自网络搜索，非知识库内容，仅供参考。\n\n{answer}",
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "province_count": len(province_codes),
            },
            tool_name=self.name,
            confidence=0.5,
        )
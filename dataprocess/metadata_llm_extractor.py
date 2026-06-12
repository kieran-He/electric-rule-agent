"""
LLM-based metadata extractor for issuer and issue_date.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from dataprocess.llm_client import call_llm_json, LLMConfig


SYSTEM_PROMPT = """你是电力市场政策文档元数据提取器。
任务：从文档文本中提取发布机构(issuer)和发布日期(issue_date)。
返回严格JSON，不要markdown。"""

USER_PROMPT_TEMPLATE = """请从以下文档文本中提取元数据，返回严格JSON：
{
  "issuer": "发布机构名称，如'冀北电力交易中心'、'陕西省发展和改革委员会'等。如果无法确定，返回null。",
  "issue_date": "发布日期，格式YYYY-MM-DD，如'2024-01-15'。如果无法确定，返回null。",
  "issuer_confidence": "高|中|低",
  "date_confidence": "高|中|低"
}

规则：
1) issuer通常是文档开头提到的发布单位，如'XX电力交易中心'、'XX发展和改革委员会'等。
2) issue_date通常是文档末尾或标题中的日期，如'2024年1月15日印发'。
3) 只返回JSON，不要markdown。
4) 无法确定时返回null。

文档文本：
__INPUT_TEXT__
"""


def parse_date_from_llm_response(date_str: str | None) -> date | None:
    """Parse date from LLM response string."""
    if not date_str:
        return None
    
    import re
    # Try YYYY-MM-DD format
    match = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", date_str)
    if match:
        year, month, day = map(int, match.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None
    
    # Try Chinese format YYYY年MM月DD日
    match = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", date_str)
    if match:
        year, month, day = map(int, match.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None
    
    return None


def extract_metadata_with_llm(*, text: str, cfg: LLMConfig, max_chars: int = 3000) -> dict[str, Any]:
    """Extract issuer and issue_date using LLM."""
    sample_text = text[:max_chars]
    
    try:
        payload = call_llm_json(
            cfg=cfg,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT_TEMPLATE.replace("__INPUT_TEXT__", sample_text),
        )
        
        issuer = payload.get("issuer")
        if issuer and isinstance(issuer, str):
            issuer = issuer.strip()
            if issuer.lower() == "null" or issuer == "":
                issuer = None
        
        issue_date_str = payload.get("issue_date")
        if issue_date_str and isinstance(issue_date_str, str):
            if issue_date_str.lower() == "null" or issue_date_str == "":
                issue_date_str = None
        
        issue_date = parse_date_from_llm_response(issue_date_str)
        
        return {
            "issuer": issuer,
            "issue_date": issue_date,
            "issuer_confidence": payload.get("issuer_confidence"),
            "date_confidence": payload.get("date_confidence"),
        }
    except Exception as e:
        print(f"[LLM Metadata] Failed: {type(e).__name__}: {str(e)[:100]}")
        return {
            "issuer": None,
            "issue_date": None,
            "issuer_confidence": None,
            "date_confidence": None,
        }
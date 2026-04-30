from __future__ import annotations

import json
from typing import Any

from dataprocess.llm_client import call_llm_json, LLMConfig
from dataprocess.schemas import RuleTagExtraction


SYSTEM_PROMPT = """你是电力市场规则标注引擎。
返回严格JSON，字段值必须在预设选项中选择，不要输出markdown。"""

USER_PROMPT_TEMPLATE = """请从文本中提取规则标签，返回严格JSON：
{
  "tags": {
    "market_type": "中长期市场|现货市场|零售市场|辅助服务市场|容量市场|储能市场|综合|null",
    "entity_type": "发电企业|售电公司|电力用户|独立储能|电源侧储能|用户侧储能|虚拟电厂|电网企业|交易中心|调度中心|null",
    "trade_cycle": "年度|月度|月内|日内|日前|实时|多日|null",
    "trade_mode": "双边协商|集中竞价|挂牌交易|滚动撮合|报量报价|报量不报价|null",
    "time_granularity": "年|月|日|小时|15分钟|时段|null",
    "action_type": "申报|报价|出清|结算|签约|考核|调度|注册|准入|null",
    "penalty_related": true|false,
    "green_power_related": true|false,
    "retail_related": true|false,
    "storage_related": true|false,
    "vpp_related": true|false
  }
}

规则：
1) 字段值必须在上述选项中选择，不能自创新词。
2) 无法识别的字段填null。
3) 布尔字段默认false。
4) 只返回JSON，不要markdown。
5) 不要臆造文本中没有的内容。

文本：
__INPUT_TEXT__
"""


STRING_FIELDS = {
    "market_type",
    "entity_type",
    "trade_cycle",
    "trade_mode",
    "time_granularity",
    "action_type",
}
BOOLEAN_FIELDS = {
    "penalty_related",
    "green_power_related",
    "retail_related",
    "storage_related",
    "vpp_related",
}

VALID_OPTIONS = {
    "market_type": ["中长期市场", "现货市场", "零售市场", "辅助服务市场", "容量市场", "储能市场", "综合", None],
    "entity_type": ["发电企业", "售电公司", "电力用户", "独立储能", "电源侧储能", "用户侧储能", "虚拟电厂", "电网企业", "交易中心", "调度中心", None],
    "trade_cycle": ["年度", "月度", "月内", "日内", "日前", "实时", "多日", None],
    "trade_mode": ["双边协商", "集中竞价", "挂牌交易", "滚动撮合", "报量报价", "报量不报价", None],
    "time_granularity": ["年", "月", "日", "小时", "15分钟", "时段", None],
    "action_type": ["申报", "报价", "出清", "结算", "签约", "考核", "调度", "注册", "准入", None],
}


class LLMTagExtractorError(RuntimeError):
    pass


class LLMTagCallError(LLMTagExtractorError):
    pass


class LLMTagParseError(LLMTagExtractorError):
    pass


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def _sanitize_llm_tags(payload: dict[str, Any]) -> RuleTagExtraction:
    tags_payload: Any = payload.get("tags", payload)
    if not isinstance(tags_payload, dict):
        raise LLMTagParseError("LLM tag payload is not an object")

    sanitized: dict[str, Any] = {}
    for key, value in tags_payload.items():
        if key not in RuleTagExtraction.model_fields:
            continue
        if key in STRING_FIELDS:
            if isinstance(value, str):
                text_value = value.strip()
                if key in VALID_OPTIONS:
                    if text_value not in VALID_OPTIONS[key]:
                        text_value = None
                sanitized[key] = text_value or None
            elif value is None:
                sanitized[key] = None
            continue
        if key in BOOLEAN_FIELDS:
            if isinstance(value, bool):
                sanitized[key] = value
            continue

    return RuleTagExtraction(**sanitized)


def extract_rule_tags_with_llm(*, text: str, cfg: LLMConfig) -> RuleTagExtraction:
    payload = call_llm_json(
        cfg=cfg,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT_TEMPLATE.replace("__INPUT_TEXT__", text),
    )
    return _sanitize_llm_tags(payload)
"""Prompt 模板和选择器模块"""
from app.prompts.prompt_templates import (
    QA_STRUCTURED_PROMPT,
    COMPARE_STRUCTURED_PROMPT,
    EXPLAIN_STRUCTURED_PROMPT,
    PROCEDURE_STRUCTURED_PROMPT,
    INTENT_KEYWORD_MAP,
)
from app.prompts.prompt_selector import PromptSelector

__all__ = [
    "QA_STRUCTURED_PROMPT",
    "COMPARE_STRUCTURED_PROMPT",
    "EXPLAIN_STRUCTURED_PROMPT",
    "PROCEDURE_STRUCTURED_PROMPT",
    "INTENT_KEYWORD_MAP",
    "PromptSelector",
]
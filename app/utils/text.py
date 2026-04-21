from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def short_excerpt(text: str, limit: int = 140) -> str:
    text = normalize_whitespace(text).replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def tokenize_len(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text))

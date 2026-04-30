from __future__ import annotations

import re
import unicodedata
from collections import Counter

from dataprocess.schemas import RawPage


PAGE_MARKER_PATTERN = re.compile(r"⟦PAGE:(\d+)⟧")

HEADER_FOOTER_PATTERNS = [
    re.compile(r"^\s*第?\s*\d+\s*页\s*/\s*共?\s*\d+\s*页?\s*$"),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*陕西电力.*?市场.*$"),
    re.compile(r"^\s*第?\s*\d+\s*页\s*$"),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*Page\s*\d+(\s*/\s*\d+)?\s*$", re.IGNORECASE),
]
TOC_NOISE_PATTERN = re.compile(r"(\.{4,}|\s{8,})\d+\s*$")
TIME_NOISE_PATTERN = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
WATERMARK_WORDS = {"交易平台", "电力市场", "普通事项"}
OCR_NOISE_REPLACEMENTS = {
    "（ ": "（",
    " ）": "）",
    "， ": "，",
    " 。": "。",
    "、 ": "、",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for source, target in OCR_NOISE_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"[|¦｜]{2,}", "|", text)
    text = re.sub(r"[·•]{2,}", "·", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_common_headers_and_footers(pages: list[str]) -> list[str]:
    if len(pages) <= 1:
        return [normalize_text(page) for page in pages]

    first_lines = Counter()
    last_lines = Counter()
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        if not lines:
            continue
        first_lines[lines[0]] += 1
        last_lines[lines[-1]] += 1

    common_headers = {line for line, count in first_lines.items() if count >= 2}
    common_footers = {line for line, count in last_lines.items() if count >= 2}

    cleaned_pages: list[str] = []
    for page in pages:
        lines = [line.strip() for line in page.splitlines()]
        filtered: list[str] = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                filtered.append("")
                continue
            if index == 0 and stripped in common_headers:
                continue
            if index == len(lines) - 1 and stripped in common_footers:
                continue
            if any(pattern.match(stripped) for pattern in HEADER_FOOTER_PATTERNS):
                continue
            filtered.append(stripped)
        cleaned_pages.append(normalize_text("\n".join(filtered)))
    return cleaned_pages


def remove_toc_noise(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if TOC_NOISE_PATTERN.search(stripped):
            continue
        if _is_watermark_noise_line(stripped):
            continue
        lines.append(stripped)
    return normalize_text("\n".join(lines))


def _is_watermark_noise_line(line: str) -> bool:
    if not line:
        return True
    if TIME_NOISE_PATTERN.match(line):
        return True
    if line in WATERMARK_WORDS:
        return True
    if re.search(r"^(20\d{2}|19\d{2})$", line):
        return True
    if re.search(r"^[\d\s:/.-]+$", line):
        return True

    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", line))
    alpha_num_chars = len(re.findall(r"[A-Za-z0-9]", line))
    if chinese_chars <= 1 and alpha_num_chars >= 4:
        return True
    return False


def clean_document_pages(pages: list[str]) -> str:
    cleaned_pages = strip_common_headers_and_footers(pages)
    merged = "\n\n".join(page for page in cleaned_pages if page)
    return remove_toc_noise(merged)


def clean_document_pages_with_markers(pages: list[RawPage]) -> str:
    cleaned_pages = strip_common_headers_and_footers([page.text for page in pages])
    marked_pages: list[str] = []
    for raw_page, cleaned_text in zip(pages, cleaned_pages):
        if cleaned_text.strip():
            marked_pages.append(f"⟦PAGE:{raw_page.page_number}⟧{cleaned_text}")
    merged = "\n\n".join(marked_pages)
    return remove_toc_noise_with_markers(merged)


def remove_toc_noise_with_markers(text: str) -> str:
    lines: list[str] = []
    current_page_marker: str | None = None
    for line in text.splitlines():
        marker_match = PAGE_MARKER_PATTERN.match(line)
        if marker_match:
            current_page_marker = marker_match.group(0)
            lines.append(line)
            continue
        stripped = line.strip()
        if TOC_NOISE_PATTERN.search(stripped):
            continue
        if _is_watermark_noise_line(stripped):
            continue
        lines.append(stripped)
    return normalize_text("\n".join(lines))


def extract_page_range_from_text(text: str) -> tuple[int, int]:
    page_numbers = [int(m.group(1)) for m in PAGE_MARKER_PATTERN.finditer(text)]
    if not page_numbers:
        return (1, 1)
    return (min(page_numbers), max(page_numbers))


def remove_page_markers(text: str) -> str:
    return PAGE_MARKER_PATTERN.sub("", text).strip()


def extract_text_between_page_markers(text: str, start_page: int, end_page: int) -> str:
    segments: list[str] = []
    collecting = False

    for line in text.splitlines():
        match = PAGE_MARKER_PATTERN.match(line)
        if match:
            current_page = int(match.group(1))
            collecting = current_page >= start_page and current_page <= end_page
            if collecting and match.end() < len(line):
                content_after_marker = line[match.end():]
                if content_after_marker.strip():
                    segments.append(content_after_marker)
            continue
        if collecting:
            segments.append(line)

    return "\n".join(segments)
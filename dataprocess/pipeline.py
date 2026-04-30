from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dataprocess.config import DocProcSettings, settings
from dataprocess.cleaner import clean_document_pages_with_markers
from dataprocess.docx_parser import parse_docx
from dataprocess.llm_client import build_llm_config
from dataprocess.llm_splitter import split_into_clauses_with_llm, _extract_rule_tags
from dataprocess.llm_tagger import extract_rule_tags_with_llm, LLMTagCallError, LLMTagParseError
from dataprocess.metadata_extractor import extract_metadata, file_sha256
from dataprocess.pdf_parser import parse_pdf
from dataprocess.schemas import (
    ClauseChunk,
    DocumentMetadata,
    ProcessedDocument,
    ProcessingStats,
    RawPage,
)


class UnsupportedFileTypeError(RuntimeError):
    pass


def _parse_pages(file_path: Path) -> list[RawPage]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(str(file_path))
    if suffix == ".docx":
        return parse_docx(str(file_path))
    raise UnsupportedFileTypeError(f"Unsupported file type: {suffix}")


def apply_document_defaults_to_clauses(clauses: list[ClauseChunk], metadata: DocumentMetadata) -> None:
    for clause in clauses:
        clause.province_code = metadata.province_code
        clause.doc_type = metadata.doc_type
        clause.doc_status = metadata.status
        clause.doc_market_type = metadata.market_type
        clause.doc_subject_scope = list(metadata.subject_scope)

        if not clause.rule_tags.market_type and metadata.market_type != "综合":
            clause.rule_tags.market_type = metadata.market_type
        if not clause.rule_tags.entity_type and metadata.subject_scope:
            clause.rule_tags.entity_type = metadata.subject_scope[0]


def build_processing_stats(clauses: list[ClauseChunk], total_pages: int) -> ProcessingStats:
    chapter_keys = {clause.chapter_no for clause in clauses if clause.chapter_no}
    section_keys = {
        f"{clause.chapter_no}-{clause.section_no}"
        for clause in clauses
        if clause.chapter_no and clause.section_no
    }
    article_keys = {
        f"{clause.chapter_no}-{clause.section_no}-{clause.article_no}"
        for clause in clauses
        if clause.article_no
    }
    coverage = {
        "market_type": 0,
        "entity_type": 0,
        "trade_cycle": 0,
        "trade_mode": 0,
        "time_granularity": 0,
        "action_type": 0,
        "penalty_related": 0,
    }

    for clause in clauses:
        tags = clause.rule_tags
        if tags.market_type:
            coverage["market_type"] += 1
        if tags.entity_type:
            coverage["entity_type"] += 1
        if tags.trade_cycle:
            coverage["trade_cycle"] += 1
        if tags.trade_mode:
            coverage["trade_mode"] += 1
        if tags.time_granularity:
            coverage["time_granularity"] += 1
        if tags.action_type:
            coverage["action_type"] += 1
        if tags.penalty_related:
            coverage["penalty_related"] += 1

    return ProcessingStats(
        total_pages=total_pages,
        total_chapters=len(chapter_keys),
        total_sections=len(section_keys),
        total_articles=len(article_keys),
        total_clauses=len(clauses),
        field_coverage=coverage,
    )


def enforce_clause_quality(
    *,
    clauses: list[ClauseChunk],
    total_pages: int,
    short_fragment_threshold: int = 20,
    drop_short_fragments: bool = False,
) -> tuple[list[ClauseChunk], dict]:
    safe_total_pages = max(1, int(total_pages))
    normalized: list[ClauseChunk] = []
    page_adjustments = 0
    dropped_short = 0
    marker_leak_count = 0
    short_fragment_indices: list[int] = []

    for clause in clauses:
        original_start = clause.page_start
        original_end = clause.page_end

        start = max(1, min(original_start, safe_total_pages))
        end = max(1, min(original_end, safe_total_pages))
        if end < start:
            end = start

        if start != original_start or end != original_end:
            page_adjustments += 1
            clause.page_start = start
            clause.page_end = end

        if "⟦PAGE:" in clause.clause_text:
            marker_leak_count += 1

        if len(clause.clause_text.strip()) < short_fragment_threshold:
            short_fragment_indices.append(len(normalized))
            if drop_short_fragments:
                dropped_short += 1
                continue

        normalized.append(clause)

    deduped: list[ClauseChunk] = []
    seen_by_page_key: set[tuple] = set()
    duplicate_same_key_removed = 0
    text_only_counter = Counter(clause.clause_text.strip() for clause in normalized if clause.clause_text.strip())
    duplicate_text_only_count = sum(count - 1 for count in text_only_counter.values() if count > 1)

    for clause in normalized:
        key = (clause.clause_text.strip(), clause.page_start, clause.page_end)
        if key in seen_by_page_key:
            duplicate_same_key_removed += 1
            continue
        seen_by_page_key.add(key)
        deduped.append(clause)

    audit = {
        "input_clause_count": len(clauses),
        "output_clause_count": len(deduped),
        "page_adjustments": page_adjustments,
        "duplicate_same_key_removed": duplicate_same_key_removed,
        "duplicate_text_only_count": duplicate_text_only_count,
        "short_fragment_threshold": short_fragment_threshold,
        "short_fragment_count": len(short_fragment_indices),
        "short_fragment_dropped": dropped_short,
        "marker_leak_count": marker_leak_count,
    }
    return deduped, audit


def process_document(
    file_path: str | Path,
    *,
    province_code_override: str | None = None,
    short_fragment_threshold: int = 20,
    drop_short_fragments: bool = False,
    config: DocProcSettings | None = None,
    checkpoint_dir: str | None = None,
) -> tuple[ProcessedDocument, dict]:
    cfg = config or settings
    path = Path(file_path)
    
    file_hash = file_sha256(path)
    metadata = extract_metadata(file_path=str(path), file_hash=file_hash, province_code_override=province_code_override)

    pages = _parse_pages(path)
    marked_text = clean_document_pages_with_markers(pages)

    llm_config = build_llm_config(cfg)
    
    clauses = split_into_clauses_with_llm(
        text=marked_text,
        doc_name=metadata.doc_name,
        source_file=str(path),
        origin_doc_id=file_hash,
        cfg=llm_config,
        max_chars_per_call=cfg.llm_max_chars_per_call,
        checkpoint_dir=checkpoint_dir,
    )

    llm_tag_fallback_count = 0
    llm_tag_fallback_reasons: Counter[str] = Counter()
    for clause in clauses:
        try:
            clause.rule_tags = extract_rule_tags_with_llm(text=clause.clause_text, cfg=llm_config)
        except Exception as exc:
            llm_tag_fallback_count += 1
            llm_tag_fallback_reasons[type(exc).__name__] += 1
            clause.rule_tags = _extract_rule_tags(clause.clause_text)

    apply_document_defaults_to_clauses(clauses, metadata)
    clauses, quality_audit = enforce_clause_quality(
        clauses=clauses,
        total_pages=len(pages),
        short_fragment_threshold=short_fragment_threshold,
        drop_short_fragments=drop_short_fragments,
    )
    stats = build_processing_stats(clauses=clauses, total_pages=len(pages))

    document = ProcessedDocument(
        metadata=metadata,
        cleaned_text=marked_text,
        clauses=clauses,
        stats=stats,
        ocr_used=False,
        processing_flags=["baseline_plus", "llm_split", "llm_tag", "page_markers"],
    )
    
    trace = {
        "pipeline": [
            "parse_pdf_or_docx",
            "clean_with_markers",
            "split_into_clauses_with_llm",
            "llm_tag_with_fallback",
            "apply_document_defaults",
            "enforce_clause_quality",
            "build_stats",
        ],
        "input_summary": {
            "total_pages": len(pages),
            "marked_text_chars": len(marked_text),
        },
        "llm_tag_fallback": {
            "fallback_count": llm_tag_fallback_count,
            "fallback_reasons": dict(llm_tag_fallback_reasons),
        },
        "quality_audit": quality_audit,
    }
    
    return document, trace


def dump_processed_document(document: ProcessedDocument, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target_file = output_path / f"{document.metadata.file_hash}.json"
    target_file.write_text(
        json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target_file
from __future__ import annotations

from dataprocess.config import DocProcSettings, settings
from dataprocess.schemas import (
    ClauseChunk,
    DocumentMetadata,
    ProcessedDocument,
    ProcessingStats,
    RawPage,
    RuleTagExtraction,
)
from dataprocess.province_mapping import (
    PROVINCE_ALIASES,
    PROVINCE_CODE_ALIASES,
    detect_province_code,
    get_province_name,
    get_all_province_codes,
)
from dataprocess.pipeline import (
    process_document,
    dump_processed_document,
    apply_document_defaults_to_clauses,
    build_processing_stats,
    enforce_clause_quality,
)
from dataprocess.metadata_extractor import (
    extract_metadata,
    file_sha256,
    extract_issuer,
)
from dataprocess.pdf_parser import parse_pdf, parse_pdf_ocr
from dataprocess.docx_parser import parse_docx
from dataprocess.cleaner import (
    clean_document_pages,
    clean_document_pages_with_markers,
    remove_page_markers,
    extract_page_range_from_text,
)
from dataprocess.llm_client import LLMConfig, build_llm_config, call_llm_json
from dataprocess.llm_splitter import split_into_clauses_with_llm
from dataprocess.llm_tagger import extract_rule_tags_with_llm
from dataprocess.ingest_to_vector import (
    ingest_processed_document_to_chroma,
    load_and_ingest_json,
)
from dataprocess.bm25_builder import ProvinceBM25Indexer

__all__ = [
    "DocProcSettings",
    "settings",
    "ClauseChunk",
    "DocumentMetadata",
    "ProcessedDocument",
    "ProcessingStats",
    "RawPage",
    "RuleTagExtraction",
    "PROVINCE_ALIASES",
    "PROVINCE_CODE_ALIASES",
    "detect_province_code",
    "get_province_name",
    "get_all_province_codes",
    "process_document",
    "dump_processed_document",
    "apply_document_defaults_to_clauses",
    "build_processing_stats",
    "enforce_clause_quality",
    "extract_metadata",
    "file_sha256",
    "extract_issuer",
    "parse_pdf",
    "parse_pdf_ocr",
    "parse_docx",
    "clean_document_pages",
    "clean_document_pages_with_markers",
    "remove_page_markers",
    "extract_page_range_from_text",
    "LLMConfig",
    "build_llm_config",
    "call_llm_json",
    "split_into_clauses_with_llm",
    "extract_rule_tags_with_llm",
    "ingest_processed_document_to_chroma",
    "load_and_ingest_json",
    "ProvinceBM25Indexer",
]
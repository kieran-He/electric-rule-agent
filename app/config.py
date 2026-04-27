from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass
class Settings:
    app_name: str = _env("APP_NAME", "shanxi-power-rule-engine")
    app_env: str = _env("APP_ENV", "dev")
    host: str = _env("HOST", "0.0.0.0")
    port: int = int(_env("PORT", "8000"))
    log_level: str = _env("LOG_LEVEL", "INFO")
    log_file: str = _env("LOG_FILE", "data/processed/app.log")
    province_default: str = _env("PROVINCE_DEFAULT", "SN")
    database_url: str = _env("DATABASE_URL", "sqlite:///./data/processed/app.db")
    chroma_path: str = _env("CHROMA_PATH", "./data/chroma")
    docs_root: str = _env("DOCS_ROOT", "./data/raw")
    embedding_model: str = _env("EMBEDDING_MODEL", "deterministic")
    reranker_model: str = _env("RERANKER_MODEL", "keyword-overlap")
    top_k: int = int(_env("TOP_K", "8"))
    vector_top_k: int = int(_env("VECTOR_TOP_K", "8"))
    keyword_top_k: int = int(_env("KEYWORD_TOP_K", "8"))
    rerank_top_k: int = int(_env("RERANK_TOP_K", "8"))
    conversation_ttl_minutes: int = int(_env("CONVERSATION_TTL_MINUTES", "120"))
    current_only_default: bool = _env("CURRENT_ONLY_DEFAULT", "true").lower() == "true"
    prefer_draft_default: bool = _env("PREFER_DRAFT_DEFAULT", "false").lower() == "true"
    tesseract_cmd: str = _env("TESSERACT_CMD", "tesseract")
    tessdata_prefix: str = _env("TESSDATA_PREFIX", "")
    ocr_enabled: bool = _env("OCR_ENABLED", "false").lower() == "true"
    ocr_min_ch_ratio: float = float(_env("OCR_MIN_CH_RATIO", "0.08"))
    ocr_max_replacement_ratio: float = float(_env("OCR_MAX_REPLACEMENT_RATIO", "0.03"))
    ocr_empty_page_threshold: float = float(_env("OCR_EMPTY_PAGE_THRESHOLD", "0.3"))
    province_confidence_threshold: float = float(_env("PROVINCE_CONFIDENCE_THRESHOLD", "0.7"))
    ingest_index_path: str = _env("INGEST_INDEX_PATH", "./data/chroma/ingest_index.json")
    hybrid_vector_top_k: int = int(_env("HYBRID_VECTOR_TOP_K", "8"))
    hybrid_bm25_top_k: int = int(_env("HYBRID_BM25_TOP_K", "8"))
    hybrid_final_top_k: int = int(_env("HYBRID_FINAL_TOP_K", "8"))
    reranker_model: str = _env("RERANKER_MODEL", "BAAI/bge-reranker-large")
    reranker_preload: bool = _env("RERANKER_PRELOAD", "true").lower() == "true"
    reranker_max_length: int = int(_env("RERANKER_MAX_LENGTH", "512"))
    bm25_k1: float = float(_env("BM25_K1", "1.5"))
    bm25_b: float = float(_env("BM25_B", "0.6"))
    query_expansion: bool = _env("QUERY_EXPANSION", "false").lower() == "true"
    query_expansion_method: str = _env("QUERY_EXPANSION_METHOD", "synonyms")
    query_expansion_max: int = int(_env("QUERY_EXPANSION_MAX", "3"))
    query_rewrite_enabled: bool = _env("QUERY_REWRITE_ENABLED", "false").lower() == "true"
    query_rewrite_min_length: int = int(_env("QUERY_REWRITE_MIN_LENGTH", "10"))
    query_rewrite_keep_original: bool = _env("QUERY_REWRITE_KEEP_ORIGINAL", "true").lower() == "true"
    feishu_webhook_url: str = _env("FEISHU_WEBHOOK_URL", "")
    feishu_alert_enabled: bool = _env("FEISHU_ALERT_ENABLED", "false").lower() == "true"
    feishu_app_id: str = _env("FEISHU_APP_ID", "")
    feishu_app_secret: str = _env("FEISHU_APP_SECRET", "")
    feishu_max_workers: int = int(_env("FEISHU_MAX_WORKERS", "10"))

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def ensure_dirs(self) -> None:
        for raw_path in [self.log_file, self.chroma_path, self.docs_root, "data/processed"]:
            path = Path(raw_path)
            target = path.parent if path.suffix else path
            target.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

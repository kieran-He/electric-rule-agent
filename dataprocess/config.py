from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    return _env(key, str(default).lower()).lower() == "true"


@dataclass
class DocProcSettings:
    llm_enabled: bool = _env_bool("DOC_PROC_LLM_ENABLED", True)
    llm_api_key: str = _env("DOC_PROC_LLM_API_KEY", "")
    llm_base_url: str = _env("DOC_PROC_LLM_BASE_URL", "https://api.minimaxi.com")
    llm_model: str = _env("DOC_PROC_LLM_MODEL", "MiniMax-M2.7")
    llm_timeout_sec: int = _env_int("DOC_PROC_LLM_TIMEOUT", 600)
    llm_max_chars_per_call: int = _env_int("DOC_PROC_LLM_MAX_CHARS_PER_CALL", 6000)
    llm_min_chars: int = _env_int("DOC_PROC_LLM_MIN_CHARS", 1200)
    
    docs_root: str = _env("DOCS_ROOT", "data/docs")
    processed_root: str = _env("DOC_PROC_PROCESSED_ROOT", "data/processed")
    chroma_path: str = _env("CHROMA_PATH", "data/chroma")
    cache_path: str = _env("DOC_PROC_CACHE_PATH", "data/cache")
    embedding_model: str = _env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    
    bm25_k1: float = float(_env("BM25_K1", "1.5"))
    bm25_b: float = float(_env("BM25_B", "0.6"))
    
    @property
    def has_llm_config(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url and self.llm_model)
    
    def ensure_dirs(self) -> None:
        for path_str in [self.docs_root, self.processed_root, self.chroma_path, self.cache_path]:
            Path(path_str).mkdir(parents=True, exist_ok=True)


settings = DocProcSettings()
settings.ensure_dirs()
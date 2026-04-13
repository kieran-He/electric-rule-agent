import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "feishu-power-policy-bot")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    chroma_path: str = os.getenv("CHROMA_PATH", "./data/chroma")
    docs_root: str = os.getenv("DOCS_ROOT", "./data/docs")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    glm_api_key: str = os.getenv("GLM_API_KEY", "")
    glm_endpoint: str = os.getenv(
        "GLM_ENDPOINT", "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    )
    glm_model: str = os.getenv("GLM_MODEL", "glm-4.5")
    feishu_token: str = os.getenv("FEISHU_VERIFICATION_TOKEN", "")
    feishu_signing_secret: str = os.getenv("FEISHU_SIGNING_SECRET", "")
    feishu_app_id: str = os.getenv("FEISHU_APP_ID", "")
    feishu_app_secret: str = os.getenv("FEISHU_APP_SECRET", "")
    province_confidence_threshold: float = float(os.getenv("PROVINCE_CONFIDENCE_THRESHOLD", "0.7"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    event_ttl_seconds: int = int(os.getenv("EVENT_TTL_SECONDS", "600"))
    ingest_index_path: str = os.getenv("INGEST_INDEX_PATH", "./data/chroma/ingest_index.json")
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "C:/Program Files/Tesseract-OCR/tesseract.exe")
    tessdata_prefix: str = os.getenv("TESSDATA_PREFIX", "")
    ocr_enabled: bool = os.getenv("OCR_ENABLED", "false").lower() == "true"
    ocr_min_ch_ratio: float = float(os.getenv("OCR_MIN_CH_RATIO", "0.08"))
    ocr_max_replacement_ratio: float = float(os.getenv("OCR_MAX_REPLACEMENT_RATIO", "0.03"))
    ocr_empty_page_threshold: float = float(os.getenv("OCR_EMPTY_PAGE_THRESHOLD", "0.3"))


settings = Settings()

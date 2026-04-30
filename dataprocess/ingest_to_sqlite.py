from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import settings as app_settings
from app.db.session import SessionLocal
from app.services.ingest.ingestion_pipeline import IngestionPipeline


def ingest_to_sqlite(
    processed_dir: Path,
    province_code: str,
    rebuild_index: bool = False,
) -> dict[str, int]:
    """Ingest processed JSON documents to SQLite database.

    This uses the IngestionPipeline which handles both SQLite and ChromaDB ingestion.

    Args:
        processed_dir: Directory containing processed JSON files
        province_code: Province code (e.g., "SN", "GS", "SD")
        rebuild_index: If True, clear existing ChromaDB collection before ingestion

    Returns:
        Dict with keys: imported_documents, imported_clauses, skipped_documents
    """
    with SessionLocal() as db:
        pipeline = IngestionPipeline(db, app_settings)
        result = pipeline.ingest_path(processed_dir, province_code, rebuild_index)
        db.commit()
    return result
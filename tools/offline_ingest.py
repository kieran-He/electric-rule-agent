import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.services.ingest.path_resolver import resolve_docs_path
from app.services.ingest.document_ingestor import DocumentIngestor
from app.core.repository import ChromaPolicyRepository


def _str_to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline ingest documents into Chroma knowledge base.")
    parser.add_argument("--kb-scope", choices=["province", "global"], required=True)
    parser.add_argument("--province-code", default=None)
    parser.add_argument("--docs-path", default=None)
    parser.add_argument("--docs-root", default=None)
    parser.add_argument("--rebuild", type=_str_to_bool, default=False)
    parser.add_argument("--dedupe", type=_str_to_bool, default=True)
    parser.add_argument("--enable-ocr", type=_str_to_bool, default=settings.ocr_enabled)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    return parser


def run_offline_ingest(
    kb_scope: str,
    province_code: Optional[str],
    docs_path: Optional[str],
    docs_root: Optional[str],
    rebuild: bool,
    dedupe: bool,
    enable_ocr: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    if kb_scope == "province" and not province_code:
        raise ValueError("province_code is required when kb_scope=province")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    resolved_docs_path = resolve_docs_path(
        kb_scope=kb_scope,
        province_code=province_code,
        docs_path=docs_path,
        docs_root=docs_root,
        default_docs_root=settings.docs_root,
    )
    repository = ChromaPolicyRepository(
        persist_directory=settings.chroma_path, embedding_model_name=settings.embedding_model
    )
    if not repository.ready:
        raise RuntimeError(f"repository not ready: {repository.init_error}")
    ingestor = DocumentIngestor(repository, index_path=settings.ingest_index_path)
    stats = ingestor.ingest_path(
        docs_path=resolved_docs_path,
        kb_scope=kb_scope,
        province_code=province_code,
        rebuild=rebuild,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_ocr=enable_ocr,
        dedupe=dedupe,
        min_ch_ratio=settings.ocr_min_ch_ratio,
        max_replacement_ratio=settings.ocr_max_replacement_ratio,
        empty_page_threshold=settings.ocr_empty_page_threshold,
    )
    return {
        "success": True,
        "kb_scope": kb_scope,
        "province_code": province_code,
        "resolved_docs_path": resolved_docs_path,
        "chroma_path": settings.chroma_path,
        "ingest_index_path": settings.ingest_index_path,
        **asdict(stats),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_offline_ingest(
            kb_scope=args.kb_scope,
            province_code=args.province_code,
            docs_path=args.docs_path,
            docs_root=args.docs_root,
            rebuild=args.rebuild,
            dedupe=args.dedupe,
            enable_ocr=args.enable_ocr,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataprocess.config import settings
from dataprocess.pipeline import process_document, dump_processed_document
from dataprocess.metadata_extractor import file_sha256
from dataprocess.province_mapping import detect_province_code, get_province_name
from dataprocess.ingest_to_vector import ingest_processed_document_to_chroma
from dataprocess.bm25_builder import ProvinceBM25Indexer


def main() -> None:
    parser = argparse.ArgumentParser(description="Process province documents")
    parser.add_argument("--province", required=True, help="Province code (e.g., GS, SD, SN)")
    parser.add_argument("--docs-root", default=None, help="Documents root directory")
    parser.add_argument("--processed-root", default=None, help="Processed output directory")
    parser.add_argument("--skip-manifest", action="store_true", help="Skip manifest check")
    parser.add_argument("--ingest", action="store_true", help="Ingest to vector store after processing")
    parser.add_argument("--timeout", type=int, default=600, help="LLM timeout in seconds")
    args = parser.parse_args()
    
    province_code = args.province.upper()
    province_name = get_province_name(province_code) or "Unknown"
    
    docs_root = Path(args.docs_root or settings.docs_root)
    if not docs_root.is_absolute():
        docs_root = PROJECT_ROOT / docs_root
    
    processed_root = Path(args.processed_root or settings.processed_root)
    if not processed_root.is_absolute():
        processed_root = PROJECT_ROOT / processed_root
    
    province_dir = docs_root / province_code
    if not province_dir.exists():
        print(f"Error: Province directory not found: {province_dir}")
        sys.exit(1)
    
    output_dir = processed_root / province_code
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Province: {province_name} ({province_code})")
    print(f"Docs directory: {province_dir}")
    print(f"Output directory: {output_dir}")
    
    manifest_path = output_dir / "_manifest.json"
    manifest: dict = {"documents": [], "created_at": datetime.now().isoformat()}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    existing_hashes = {d.get("file_hash") for d in manifest.get("documents", [])}
    
    files = list(province_dir.glob("*.pdf")) + list(province_dir.glob("*.docx"))
    
    print(f"Found {len(files)} documents")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for file_path in files:
        file_hash = file_sha256(file_path)
        
        if not args.skip_manifest and file_hash in existing_hashes:
            print(f"Skipping (already processed): {file_path.name}")
            skipped_count += 1
            continue
        
        print(f"\nProcessing: {file_path.name}")
        print(f"Started at: {datetime.now().isoformat()}")
        
        try:
            processed, trace = process_document(
                file_path,
                province_code_override=province_code,
            )
            
            target_file = dump_processed_document(processed, output_dir)
            
            doc_record = {
                "file_name": file_path.name,
                "file_hash": file_hash,
                "output_file": str(target_file),
                "province_code": province_code,
                "province_name": province_name,
                "total_clauses": processed.stats.total_clauses,
                "total_pages": processed.stats.total_pages,
                "processing_flags": processed.processing_flags,
                "processed_at": datetime.now().isoformat(),
            }
            manifest.setdefault("documents", []).append(doc_record)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            
            trace_path = output_dir / f"{file_hash}_trace.json"
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
            
            processed_count += 1
            print(f"Completed: {processed.stats.total_clauses} clauses, {processed.stats.total_pages} pages")
            
            if args.ingest:
                ingest_result = ingest_processed_document_to_chroma(processed, province_code=province_code)
                print(f"Ingested: {ingest_result.get('ingested_chunks', 0)} chunks")
            
        except Exception as exc:
            error_count += 1
            error_record = {
                "file_name": file_path.name,
                "file_hash": file_hash,
                "error": str(exc),
                "failed_at": datetime.now().isoformat(),
            }
            manifest.setdefault("failed_documents", []).append(error_record)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Error: {exc}")
    
    print(f"\n{'='*50}")
    print(f"Summary for {province_code}:")
    print(f"  Total files: {len(files)}")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    
    if args.ingest and processed_count > 0:
        print(f"\nBuilding BM25 index...")
        bm25_indexer = ProvinceBM25Indexer(province_code, output_dir)
        bm25_docs = bm25_indexer.build_index()
        print(f"BM25 index: {bm25_docs} documents")


if __name__ == "__main__":
    main()
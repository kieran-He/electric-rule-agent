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
from dataprocess.province_mapping import detect_province_code, get_province_name, get_all_province_codes
from dataprocess.ingest_to_vector import ingest_processed_document_to_chroma, load_and_ingest_json
from dataprocess.ingest_to_sqlite import ingest_to_sqlite
from dataprocess.bm25_builder import ProvinceBM25Indexer


def cmd_process(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detected_code = detect_province_code(input_path)
    province_code = args.province_code or detected_code
    province_name = get_province_name(province_code) or "Unknown"
    
    print(f"Processing: {input_path.name}")
    print(f"Province: {province_name} ({province_code})")
    print(f"Output: {output_dir}")
    
    manifest_path = output_dir / "_manifest.json"
    manifest: dict = {"documents": [], "created_at": datetime.now().isoformat()}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    file_hash = file_sha256(input_path)
    
    existing = next((d for d in manifest.get("documents", []) if d.get("file_hash") == file_hash), None)
    if existing:
        print(f"Already processed: {existing.get('output_file')}")
        sys.exit(0)
    
    try:
        print(f"Starting processing at {datetime.now().isoformat()}")
        checkpoint_dir = None if args.no_checkpoint else str(output_dir)
        processed, trace = process_document(
            input_path,
            province_code_override=args.province_code,
            short_fragment_threshold=args.short_fragment_threshold,
            drop_short_fragments=args.drop_short_fragments,
            checkpoint_dir=checkpoint_dir,
        )
        
        target_file = dump_processed_document(processed, output_dir)
        
        doc_record = {
            "file_name": input_path.name,
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
        
        print(f"Completed: {target_file}")
        print(f"Total clauses: {processed.stats.total_clauses}")
        print(f"Total pages: {processed.stats.total_pages}")
        
        if args.ingest:
            print("Ingesting to vector store...")
            result = ingest_processed_document_to_chroma(processed, province_code=province_code)
            print(f"Ingested chunks: {result.get('ingested_chunks', 0)}")
            
            bm25_indexer = ProvinceBM25Indexer(province_code, output_dir)
            bm25_docs = bm25_indexer.build_index()
            print(f"BM25 index built: {bm25_docs} documents")
    
    except Exception as exc:
        error_record = {
            "file_name": input_path.name,
            "file_hash": file_hash,
            "error": str(exc),
            "failed_at": datetime.now().isoformat(),
        }
        manifest.setdefault("failed_documents", []).append(error_record)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Error: {exc}")
        raise


def cmd_ingest(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    
    if input_path.is_file() and input_path.suffix.lower() == ".json":
        print(f"Ingesting: {input_path}")
        result = load_and_ingest_json(input_path)
        print(f"Ingested chunks: {result.get('ingested_chunks', 0)}")
    elif input_path.is_dir():
        province_code = args.province_code or detect_province_code(input_path)
        print(f"Ingesting directory: {input_path}")
        print(f"Province: {province_code}")
        
        json_files = [f for f in input_path.glob("*.json") if not f.name.startswith("_")]
        total_chunks = 0
        for json_file in json_files:
            try:
                result = load_and_ingest_json(json_file, province_code=province_code)
                total_chunks += result.get("ingested_chunks", 0)
                print(f"  {json_file.name}: {result.get('ingested_chunks', 0)} chunks")
            except Exception as exc:
                print(f"  {json_file.name}: Error - {exc}")
        
        print(f"Total ingested: {total_chunks} chunks")
        
        bm25_indexer = ProvinceBM25Indexer(province_code, input_path)
        bm25_docs = bm25_indexer.build_index()
        print(f"BM25 index built: {bm25_docs} documents")
    else:
        print(f"Error: Input must be a JSON file or directory")
        sys.exit(1)


def cmd_province(args: argparse.Namespace) -> None:
    docs_root = Path(args.docs_root or settings.docs_root)
    if not docs_root.is_absolute():
        docs_root = PROJECT_ROOT / docs_root
    
    processed_root = Path(args.processed_root or settings.processed_root)
    if not processed_root.is_absolute():
        processed_root = PROJECT_ROOT / processed_root
    
    if args.all:
        province_codes = get_all_province_codes()
        for code in province_codes:
            province_dir = docs_root / code
            if province_dir.exists() and province_dir.is_dir():
                print(f"\n{'='*50}")
                print(f"Processing province: {get_province_name(code)} ({code})")
                process_province_directory(
                    province_dir,
                    processed_root / code,
                    code,
                    args.skip_manifest,
                )
    elif args.province_code:
        code = args.province_code.upper()
        province_dir = docs_root / code
        if not province_dir.exists():
            print(f"Error: Province directory not found: {province_dir}")
            sys.exit(1)
        
        output_dir = processed_root / code
        print(f"Processing province: {get_province_name(code)} ({code})")
        process_province_directory(
            province_dir,
            output_dir,
            code,
            args.skip_manifest,
        )
    else:
        print("Error: Specify --province-code or --all")
        sys.exit(1)


def process_province_directory(
    docs_dir: Path,
    output_dir: Path,
    province_code: str,
    skip_manifest: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = output_dir / "_manifest.json"
    manifest: dict = {"documents": [], "created_at": datetime.now().isoformat()}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    existing_hashes = {d.get("file_hash") for d in manifest.get("documents", [])}
    
    files = list(docs_dir.glob("*.pdf")) + list(docs_dir.glob("*.docx"))
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for file_path in files:
        file_hash = file_sha256(file_path)
        
        if not skip_manifest and file_hash in existing_hashes:
            print(f"  Skipping (already processed): {file_path.name}")
            skipped_count += 1
            continue
        
        print(f"  Processing: {file_path.name}")
        
        try:
            processed, trace = process_document(
                file_path,
                province_code_override=province_code,
                checkpoint_dir=str(output_dir),
            )
            
            target_file = dump_processed_document(processed, output_dir)
            
            doc_record = {
                "file_name": file_path.name,
                "file_hash": file_hash,
                "output_file": str(target_file),
                "province_code": province_code,
                "province_name": get_province_name(province_code),
                "total_clauses": processed.stats.total_clauses,
                "total_pages": processed.stats.total_pages,
                "processing_flags": processed.processing_flags,
                "processed_at": datetime.now().isoformat(),
            }
            manifest.setdefault("documents", []).append(doc_record)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            
            processed_count += 1
            print(f"    Done: {processed.stats.total_clauses} clauses, {processed.stats.total_pages} pages")
            
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
            print(f"    Error: {exc}")
    
    print(f"\nSummary for {province_code}:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")


def full_ingest_province(
    province_code: str,
    docs_root: Path,
    processed_root: Path,
    skip_process: bool = False,
    force: bool = False,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Execute full ingestion pipeline for a province.

    Args:
        province_code: Province code (e.g., "SN", "GS")
        docs_root: Root directory for source documents
        processed_root: Root directory for processed JSON files
        skip_process: Skip document processing, only ingest
        force: Force reprocessing even if already processed
        rebuild: Clear vector store and rebuild index

    Returns:
        Dict with processing and ingestion results
    """
    province_code = province_code.upper()
    province_name = get_province_name(province_code) or "Unknown"
    province_dir = docs_root / province_code
    output_dir = processed_root / province_code

    result: dict[str, Any] = {
        "province_code": province_code,
        "province_name": province_name,
        "processed": 0,
        "process_errors": 0,
        "chroma_chunks": 0,
        "sqlite_docs": 0,
        "sqlite_clauses": 0,
        "sqlite_skipped": 0,
        "bm25_docs": 0,
    }

    print(f"\n{'='*50}")
    print(f"Full ingest: {province_name} ({province_code})")
    print(f"{'='*50}")

    if not skip_process:
        if not province_dir.exists():
            print(f"Warning: Source directory not found: {province_dir}")
        else:
            print(f"\n[1/4] Processing documents...")
            output_dir.mkdir(parents=True, exist_ok=True)

            manifest_path = output_dir / "_manifest.json"
            manifest: dict = {"documents": [], "created_at": datetime.now().isoformat()}
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            existing_hashes = set() if force else {d.get("file_hash") for d in manifest.get("documents", [])}

            files = list(province_dir.glob("*.pdf")) + list(province_dir.glob("*.docx"))

            for file_path in files:
                file_hash = file_sha256(file_path)

                if file_hash in existing_hashes:
                    print(f"  Skipping (already processed): {file_path.name}")
                    continue

                print(f"  Processing: {file_path.name}")

                try:
                    processed, trace = process_document(
                        file_path,
                        province_code_override=province_code,
                        checkpoint_dir=str(output_dir),
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

                    result["processed"] += 1
                    print(f"    Done: {processed.stats.total_clauses} clauses, {processed.stats.total_pages} pages")

                except Exception as exc:
                    result["process_errors"] += 1
                    error_record = {
                        "file_name": file_path.name,
                        "file_hash": file_hash,
                        "error": str(exc),
                        "failed_at": datetime.now().isoformat(),
                    }
                    manifest.setdefault("failed_documents", []).append(error_record)
                    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"    Error: {exc}")
    else:
        print(f"\n[1/4] Skipping document processing (--skip-process)")

    if not output_dir.exists():
        print(f"Warning: Processed directory not found: {output_dir}")
        return result

    print(f"\n[2/4] Ingesting to ChromaDB...")
    json_files = [f for f in output_dir.glob("*.json") if not f.name.startswith("_")]
    total_chunks = 0
    for json_file in json_files:
        try:
            ingest_result = load_and_ingest_json(json_file, province_code=province_code)
            total_chunks += ingest_result.get("ingested_chunks", 0)
            print(f"  {json_file.name}: {ingest_result.get('ingested_chunks', 0)} chunks")
        except Exception as exc:
            print(f"  {json_file.name}: Error - {exc}")
    result["chroma_chunks"] = total_chunks
    print(f"  Total ChromaDB chunks: {total_chunks}")

    print(f"\n[3/4] Ingesting to SQLite...")
    try:
        sqlite_result = ingest_to_sqlite(output_dir, province_code, rebuild_index=rebuild)
        result["sqlite_docs"] = sqlite_result.get("imported_documents", 0)
        result["sqlite_clauses"] = sqlite_result.get("imported_clauses", 0)
        result["sqlite_skipped"] = sqlite_result.get("skipped_documents", 0)
        print(f"  SQLite: {result['sqlite_docs']} docs, {result['sqlite_clauses']} clauses, {result['sqlite_skipped']} skipped")
    except Exception as exc:
        print(f"  SQLite error: {exc}")

    print(f"\n[4/4] Building BM25 index...")
    try:
        bm25_indexer = ProvinceBM25Indexer(province_code, output_dir)
        bm25_docs = bm25_indexer.build_index()
        result["bm25_docs"] = bm25_docs
        print(f"  BM25 index: {bm25_docs} documents")
    except Exception as exc:
        print(f"  BM25 error: {exc}")

    print(f"\n{'='*50}")
    print(f"Summary for {province_code}:")
    print(f"  Documents processed: {result['processed']}")
    print(f"  Processing errors: {result['process_errors']}")
    print(f"  ChromaDB chunks: {result['chroma_chunks']}")
    print(f"  SQLite docs: {result['sqlite_docs']} (+ {result['sqlite_skipped']} skipped)")
    print(f"  SQLite clauses: {result['sqlite_clauses']}")
    print(f"  BM25 indexed: {result['bm25_docs']}")
    print(f"{'='*50}")

    return result


def cmd_full_ingest(args: argparse.Namespace) -> None:
    docs_root = Path(args.docs_root or settings.docs_root)
    if not docs_root.is_absolute():
        docs_root = PROJECT_ROOT / docs_root

    processed_root = Path(args.processed_root or settings.processed_root)
    if not processed_root.is_absolute():
        processed_root = PROJECT_ROOT / processed_root

    if args.all:
        province_codes = get_all_province_codes()
        results = []
        for code in province_codes:
            province_dir = docs_root / code
            if province_dir.exists() and province_dir.is_dir():
                result = full_ingest_province(
                    province_code=code,
                    docs_root=docs_root,
                    processed_root=processed_root,
                    skip_process=args.skip_process,
                    force=args.force,
                    rebuild=args.rebuild,
                )
                results.append(result)
        print(f"\n{'='*50}")
        print("Batch processing complete:")
        for r in results:
            print(f"  {r['province_code']}: {r['processed']} processed, {r['chroma_chunks']} chunks, {r['sqlite_docs']} docs")
    elif args.province_code:
        full_ingest_province(
            province_code=args.province_code,
            docs_root=docs_root,
            processed_root=processed_root,
            skip_process=args.skip_process,
            force=args.force,
            rebuild=args.rebuild,
        )
    else:
        print("Error: Specify --province-code or --all")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Document processing pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    proc_parser = subparsers.add_parser("process", help="Process a single document")
    proc_parser.add_argument("--input", required=True, help="Input PDF/DOCX file path")
    proc_parser.add_argument("--output", default="data/processed", help="Output directory")
    proc_parser.add_argument("--province-code", default=None, help="Override province code")
    proc_parser.add_argument("--short-fragment-threshold", type=int, default=20)
    proc_parser.add_argument("--drop-short-fragments", action="store_true")
    proc_parser.add_argument("--ingest", action="store_true", help="Ingest to vector store after processing")
    proc_parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint saving (default: enabled)")
    
    ingest_parser = subparsers.add_parser("ingest", help="Ingest processed JSON to vector store")
    ingest_parser.add_argument("--input", required=True, help="Input JSON file or directory")
    ingest_parser.add_argument("--province-code", default=None, help="Province code")
    
    prov_parser = subparsers.add_parser("province", help="Process all documents in a province")
    prov_parser.add_argument("--province-code", default=None, help="Province code (e.g., GS, SD)")
    prov_parser.add_argument("--all", action="store_true", help="Process all provinces")
    prov_parser.add_argument("--docs-root", default=None, help="Documents root directory")
    prov_parser.add_argument("--processed-root", default=None, help="Processed output directory")
    prov_parser.add_argument("--skip-manifest", action="store_true", help="Skip manifest check (reprocess all)")

    full_ingest_parser = subparsers.add_parser("full-ingest", help="Full pipeline: process + ChromaDB + SQLite + BM25")
    full_ingest_parser.add_argument("--province-code", default=None, help="Province code (e.g., GS, SD)")
    full_ingest_parser.add_argument("--all", action="store_true", help="Process all provinces")
    full_ingest_parser.add_argument("--docs-root", default=None, help="Documents root directory")
    full_ingest_parser.add_argument("--processed-root", default=None, help="Processed output directory")
    full_ingest_parser.add_argument("--skip-process", action="store_true", help="Skip document processing (only ingest)")
    full_ingest_parser.add_argument("--force", action="store_true", help="Force reprocessing of already processed documents")
    full_ingest_parser.add_argument("--rebuild", action="store_true", help="Clear vector store and rebuild index")

    args = parser.parse_args()
    
    if args.command == "process":
        cmd_process(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "province":
        cmd_province(args)
    elif args.command == "full-ingest":
        cmd_full_ingest(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
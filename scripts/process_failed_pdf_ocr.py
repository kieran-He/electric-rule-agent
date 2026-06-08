import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataprocess.pdf_parser import parse_pdf_ocr
from dataprocess.cleaner import clean_document_pages_with_markers
from dataprocess.metadata_extractor import extract_metadata, file_sha256
from dataprocess.schemas import ClauseChunk, ProcessedDocument, ProcessingStats
from process_pdf_rule_based import split_text_rule_based
from dataprocess.llm_splitter import _extract_rule_tags
from app.config import settings as app_settings
import sqlite3
import time
import json

FAILED_PDFS = [
    ("CQ", "2026年重庆绿色电力交易指导.pdf"),
    ("CQ", "重庆电力现货市场结算实施细则.pdf"),
    ("CQ", "重庆电力零售市场信息披露实施细则公告.pdf"),
    ("FJ", "福建电力中长期市场交易规则办法（试行）.pdf"),
    ("FJ", "福建电力中长期市场基本规则（2025年修订版）.pdf"),
    ("JB", "南方区域分布式电源参与市场交易方案.pdf"),
    ("JL", "吉林省电力中长期市场实施细则（试行）.pdf"),
    ("JN", "内蒙古自治区发展和改革委员会能源局关于蒙西电网电力中长期市场交易规则及实时调整细则的通知（内能源局综〔2025〕1444号）.pdf"),
    ("SN", "陕西省电力中长期市场实施细则.pdf"),
]

TESSERACT_CMD = app_settings.tesseract_cmd
TESSDATA_PREFIX = app_settings.tessdata_prefix
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)
checkpoint_file = processed_dir / "ocr_checkpoint.json"

print(f"OCR config: Tesseract={TESSERACT_CMD}")
print(f"Processing {len(FAILED_PDFS)} failed PDFs...")
start_time = time.time()
total_docs = 0
total_clauses = 0
total_pages_sum = 0
for prov, filename in FAILED_PDFS:
    pdf_path = Path("data/docs") / prov / filename
    
    if not pdf_path.exists():
        prov_dir = Path("data/docs") / prov
        actual_files = list(prov_dir.glob("*.pdf"))
        if actual_files:
            pdf_path = actual_files[0]
            filename = pdf_path.name
        else:
            print(f"  File not found in {prov}, skipping")
            continue
    
    key = f"{prov}:{filename}"
    print(f"\n[{total_docs+1}/{len(FAILED_PDFS)}] {prov}/{filename}")
    print(f"  Size: {pdf_path.stat().st_size / 1024:.1f} KB")
    checkpoint = None
    if checkpoint_file.exists():
        checkpoint = json.loads(checkpoint_file.read_text(encoding='utf-8'))
        processed_set = set(checkpoint.get('processed_files', []))
        if key in processed_set:
            print(f"  Already processed, skipping")
            total_docs += 1
            continue
    else:
        checkpoint = {
            "processed_files": [],
            "failed_files": [],
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
    print(f"  Running OCR...")
    try:
        pages = parse_pdf_ocr(
            str(pdf_path),
            lang="chi_sim+eng",
            dpi=300,
            tesseract_cmd=TESSERACT_CMD,
            tessdata_dir=TESSDATA_PREFIX,
        )
        num_pages = len(pages)
        print(f"  OCR success: {num_pages} pages")
        marked_text = clean_document_pages_with_markers(pages)
        file_hash = file_sha256(pdf_path)
        metadata = extract_metadata(
            file_path=str(pdf_path),
            file_hash=file_hash,
            province_code_override=prov,
            doc_text=marked_text[:500],
        )
        chunks = split_text_rule_based(marked_text, chunk_size=400, chunk_overlap=60)
        clauses = []
        for i, chunk in enumerate(chunks):
            rule_tags = _extract_rule_tags(chunk["text"])
            clause = ClauseChunk(
                doc_name=metadata.doc_name,
                source_file=str(pdf_path),
                origin_doc_id=file_hash,
                province_code=prov,
                doc_type=metadata.doc_type,
                doc_status=metadata.status,
                doc_issuer=metadata.issuer,
                title_path=f"chunk_{i+1}",
                clause_text=chunk["text"],
                clause_summary=chunk["text"][:50] if len(chunk["text"]) > 50 else chunk["text"],
                page_start=1,
                page_end=num_pages,
                token_count=max(1, len(chunk["text"]) // 2),
                rule_tags=rule_tags,
            )
            clauses.append(clause)
        output_prov_dir = processed_dir / prov
        output_prov_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_prov_dir / f"{file_hash}.json"
        doc = ProcessedDocument(
            metadata=metadata,
            cleaned_text=marked_text,
            clauses=clauses,
            stats=ProcessingStats(total_pages=num_pages, total_clauses=len(clauses)),
            processing_flags=["ocr", "rule_based_split", f"pages_{num_pages}"],
        )
        output_file.write_text(
            json.dumps(doc.model_dump(mode='json'), ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        checkpoint["processed_files"].append(key)
        checkpoint_file.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        print(f"  Saved: {output_file}")
        total_docs += 1
        total_clauses += len(clauses)
        total_pages_sum += num_pages
    except Exception as e:
        checkpoint["failed_files"].append({"key": key, "error": str(e)[:200]})
        checkpoint_file.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        print(f"  OCR failed: {type(e).__name__}")
        continue

elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"OCR Processing Complete")
print(f"{'='*60}")
print(f"Success: {total_docs} docs")
print(f"Failed: {len(FAILED_PDFS) - total_docs} docs")
print(f"Total pages: {total_pages_sum}")
print(f"Total clauses: {total_clauses}")
print(f"Time: {elapsed:.1f}s")

checkpoint_file.unlink()

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.services.ingest.ingestion_pipeline import IngestionPipeline
print("\nImporting to database...")
engine = create_engine(app_settings.database_url)
Session = sessionmaker(bind=engine)
imported_docs = 0
imported_clauses = 0
for prov, _ in FAILED_PDFS:
    prov_dir = processed_dir / prov
    if prov_dir.exists():
        json_files = list(prov_dir.glob("*.json"))
        if json_files:
            with Session() as db:
                pipeline = IngestionPipeline(db=db, settings=app_settings)
                result = pipeline.ingest_path(
                    path=prov_dir,
                    province_code=prov,
                    rebuild_index=False,
                )
                db.commit()
                imported_docs += result['imported_documents']
                imported_clauses += result['imported_clauses']
                print(f"  {prov}: {result['imported_documents']} docs, {result['imported_clauses']} clauses")

conn = sqlite3.connect('data/processed/app.db')
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM document')
final_docs = c.fetchone()[0]
c.execute('SELECT COUNT(*) FROM clause')
final_clauses = c.fetchone()[0]
conn.close()
print(f"\nFinal database state:")
print(f"  Documents: {final_docs}")
print(f"  Clauses: {final_clauses}")
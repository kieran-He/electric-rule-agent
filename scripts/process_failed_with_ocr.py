"""
处理解析失败的PDF文档（使用OCR）

用法：python scripts/process_failed_with_ocr.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import sqlite3
from dataprocess.pdf_parser import parse_pdf_ocr, DocumentParseError
from dataprocess.cleaner import clean_document_pages_with_markers
from dataprocess.metadata_extractor import extract_metadata, file_sha256
from dataprocess.schemas import ClauseChunk, ProcessedDocument, ProcessingStats
from dataprocess.llm_splitter import _extract_rule_tags
from process_pdf_rule_based import split_text_rule_based
from app.config import settings
from app.services.ingest.ingestion_pipeline import IngestionPipeline
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import time

# Failed PDFs from checkpoint
FAILED_PDFS = [
    ("CQ", "2026年重庆绿色电力交易指南.pdf"),
    ("CQ", "重庆电力现货市场结算实施细则.pdf"),
    ("CQ", "重庆电力现货市场零售侧管理实施细则公告.pdf"),
    ("FJ", "福建省电力中长期市场交易管理办法（试行）.pdf"),
    ("FJ", "福建省电力中长期市场中长期交易规则2025年修订版）.pdf"),
    ("JB", "华北分布式发电参与电力市场交易实施细则.pdf"),
    ("JL", "吉林省电力中长期市场实施细则（试行）.pdf"),
    ("JN", "河北省发展改革委华北能源监管局关于印发蒙西地区电力市场交易规则暨实时执行细则的通知冀发改运行〔2025〕1444号）.pdf"),
    ("SN", "陕西省电力中长期市场实施细则.pdf"),
]

PROCESSED_DIR = Path("data/processed")
DB_PATH = "data/processed/app.db"


def process_with_ocr(pdf_path: Path, province_code: str) -> tuple[bool, str]:
    """使用OCR处理单个PDF"""
    try:
        print(f"  OCR processing...")
        start = time.time()
        
        # OCR parse
        pages = parse_pdf_ocr(
            str(pdf_path),
            lang="chi_sim+eng",
            dpi=300,
            tesseract_cmd=settings.tesseract_cmd,
            tessdata_dir=settings.tessdata_prefix,
        )
        ocr_time = time.time() - start
        print(f"  OCR done: {len(pages)} pages, {ocr_time:.1f}s")
        
        # Clean text
        marked_text = clean_document_pages_with_markers(pages)
        
        # Extract metadata
        file_hash = file_sha256(pdf_path)
        metadata = extract_metadata(
            str(pdf_path),
            file_hash,
            province_code,
            marked_text[:500],
        )
        
        # Rule-based split
        raw_chunks = split_text_rule_based(marked_text, chunk_size=400, chunk_overlap=60)
        
        # Build clauses
        clauses = []
        for i, chunk in enumerate(raw_chunks):
            rule_tags = _extract_rule_tags(chunk["text"])
            clause = ClauseChunk(
                doc_name=metadata.doc_name,
                source_file=str(pdf_path),
                origin_doc_id=file_hash,
                province_code=province_code,
                doc_type=metadata.doc_type,
                doc_status=metadata.status,
                doc_issuer=metadata.issuer,
                title_path=f"chunk_{i+1}",
                clause_text=chunk["text"],
                clause_summary=chunk["text"][:50] if len(chunk["text"]) > 50 else chunk["text"],
                page_start=1,
                page_end=len(pages),
                token_count=max(1, len(chunk["text"]) // 2),
                rule_tags=rule_tags,
            )
            clauses.append(clause)
        
        # Build ProcessedDocument
        doc = ProcessedDocument(
            metadata=metadata,
            cleaned_text=marked_text,
            clauses=clauses,
            stats=ProcessingStats(total_pages=len(pages), total_clauses=len(clauses)),
            processing_flags=["ocr", "rule_based_split", f"pages_{len(pages)}"],
            ocr_used=True,
        )
        
        # Save JSON
        output_dir = PROCESSED_DIR / province_code
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{file_hash}.json"
        output_file.write_text(
            json.dumps(doc.model_dump(mode='json'), ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        
        total_time = time.time() - start
        return True, f"OCR成功: {len(pages)}页, {len(clauses)}条, issuer={metadata.issuer}, 耗时{total_time:.1f}s"
        
    except Exception as e:
        return False, f"OCR失败: {type(e).__name__}: {str(e)[:100]}"


def main():
    print("=" * 60)
    print("OCR处理失败文档")
    print("=" * 60)
    
    docs_root = Path("data/docs")
    stats = {"success": 0, "failed": 0, "total_clauses": 0}
    
    engine = create_engine(settings.database_url)
    Session = sessionmaker(bind=engine)
    
    for province, filename in FAILED_PDFS:
        pdf_path = docs_root / province / filename
        
        if not pdf_path.exists():
            print(f"\n[{province}] {filename}")
            print(f"  文件不存在")
            stats["failed"] += 1
            continue
        
        print(f"\n[{province}] {filename}")
        print(f"  大小: {pdf_path.stat().st_size / 1024:.1f} KB")
        
        success, msg = process_with_ocr(pdf_path, province)
        
        if success:
            stats["success"] += 1
            print(f"  {msg}")
        else:
            stats["failed"] += 1
            print(f"  {msg}")
    
    # Import to database
    print("\n" + "=" * 60)
    print("入库到数据库")
    print("=" * 60)
    
    with Session() as db:
        pipeline = IngestionPipeline(db=db, settings=settings)
        
        for province, _ in FAILED_PDFS:
            province_dir = PROCESSED_DIR / province
            if province_dir.exists():
                json_files = list(province_dir.glob("*.json"))
                # Check if any new files (OCR processed)
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT file_hash FROM document WHERE province_code = ?", (province,))
                existing_hashes = set(row[0] for row in cursor.fetchall())
                conn.close()
                
                new_files = [f for f in json_files if f.stem not in existing_hashes]
                if new_files:
                    print(f"\n入库 {province}: {len(new_files)} 个新文档")
                    try:
                        result = pipeline.ingest_path(
                            path=province_dir,
                            province_code=province,
                            rebuild_index=False
                        )
                        db.commit()
                        print(f"  导入文档: {result['imported_documents']}")
                        print(f"  导入条款: {result['imported_clauses']}")
                    except Exception as e:
                        print(f"  入库失败: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"OCR成功: {stats['success']}")
    print(f"OCR失败: {stats['failed']}")
    
    # Final database stats
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM document")
    doc_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM clause")
    clause_count = cursor.fetchone()[0]
    conn.close()
    
    print(f"\n数据库总文档: {doc_count}")
    print(f"数据库总条款: {clause_count}")


if __name__ == "__main__":
    main()
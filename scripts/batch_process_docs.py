"""
批量文档处理脚本 - 支持中断和重启

功能：
1. 清空数据库和切分数据
2. 按省份处理所有PDF文档
3. 支持checkpoint机制，可中断重启

用法：
python scripts/batch_process_docs.py [--clean] [--province SX] [--resume]

参数：
--clean    清空数据库和切分数据（第一次运行时使用）
--province 只处理指定省份（可选，默认处理所有省份）
--resume   从上次中断的位置继续（默认行为）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import sqlite3
from datetime import datetime
from collections import Counter

from dataprocess.pdf_parser import parse_pdf
from dataprocess.cleaner import clean_document_pages_with_markers
from dataprocess.llm_splitter import _extract_rule_tags
from process_pdf_rule_based import split_text_rule_based
from dataprocess.metadata_extractor import extract_metadata, file_sha256
from dataprocess.schemas import ClauseChunk, ProcessedDocument, ProcessingStats
from app.services.ingest.ingestion_pipeline import IngestionPipeline
from app.config import settings


CHECKPOINT_FILE = "data/processed/batch_checkpoint.json"
DB_PATH = "data/processed/app.db"
PROCESSED_DIR = Path("data/processed")


def clean_database():
    """清空数据库表数据"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    tables = ['clause', 'rule_tag', 'document']
    for table in tables:
        cursor.execute(f'DELETE FROM {table}')
        print(f'  清空表: {table}')
    
    conn.commit()
    conn.close()
    print('数据库清空完成')


def clean_processed_data():
    """清空切分数据目录"""
    import shutil
    
    dirs_to_clean = ['data/processed/SX', 'data/processed/AH', 'data/processed/CQ']
    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f'  清空目录: {dir_name}')
    
    # 清空chroma向量库
    chroma_path = Path(settings.chroma_path)
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        print(f'  清空向量库: {chroma_path}')
    
    # 删除checkpoint文件
    checkpoint_path = Path(CHECKPOINT_FILE)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f'  删除checkpoint文件')
    
    print('切分数据清空完成')


def load_checkpoint() -> dict:
    """加载checkpoint"""
    checkpoint_path = Path(CHECKPOINT_FILE)
    if checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        print(f'加载checkpoint: 已处理 {data["total_processed"]} 个文档')
        return data
    return {
        "processed_files": [],
        "failed_files": [],
        "total_processed": 0,
        "total_failed": 0,
        "last_province": None,
        "last_file": None,
        "started_at": datetime.now().isoformat(),
    }


def save_checkpoint(checkpoint: dict):
    """保存checkpoint"""
    checkpoint_path = Path(CHECKPOINT_FILE)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["updated_at"] = datetime.now().isoformat()
    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding='utf-8')


def get_province_docs(base_dir: Path = Path("data/docs")) -> dict[str, list[Path]]:
    """获取各省文档列表"""
    provinces = {}
    for province_dir in base_dir.iterdir():
        if province_dir.is_dir():
            pdf_files = list(province_dir.glob("*.pdf"))
            if pdf_files:
                provinces[province_dir.name] = sorted(pdf_files)
    return provinces


def process_single_doc(pdf_path: Path, province_code: str) -> tuple[bool, str]:
    """处理单个文档（纯规则切分，快速）"""
    try:
        file_hash = file_sha256(pdf_path)
        
        # Parse PDF
        pages = parse_pdf(str(pdf_path))
        total_pages = len(pages)
        
        # Clean text
        marked_text = clean_document_pages_with_markers(pages)
        
        # Extract metadata with issuer
        metadata = extract_metadata(
            file_path=str(pdf_path),
            file_hash=file_hash,
            province_code_override=province_code,
            doc_text=marked_text[:500],
        )
        
        # Rule-based split
        raw_chunks = split_text_rule_based(marked_text, chunk_size=400, chunk_overlap=60)
        
        # Build ClauseChunk list
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
                page_end=total_pages,
                token_count=max(1, len(chunk["text"]) // 2),
                rule_tags=rule_tags,
            )
            clauses.append(clause)
        
        # Build ProcessedDocument
        doc = ProcessedDocument(
            metadata=metadata,
            cleaned_text=marked_text,
            clauses=clauses,
            stats=ProcessingStats(total_pages=total_pages, total_clauses=len(clauses)),
            processing_flags=["rule_based_split", f"pages_{total_pages}"],
        )
        
        # Save JSON
        output_dir = PROCESSED_DIR / province_code
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{file_hash}.json"
        output_file.write_text(
            json.dumps(doc.model_dump(mode='json'), ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        
        return True, f"成功: {total_pages}页, {len(clauses)}条, issuer={metadata.issuer}"
    except Exception as e:
        return False, f"失败: {type(e).__name__}: {str(e)[:100]}"


def main():
    parser = argparse.ArgumentParser(description='批量处理文档')
    parser.add_argument('--clean', action='store_true', help='清空数据库和切分数据')
    parser.add_argument('--province', type=str, help='只处理指定省份')
    parser.add_argument('--resume', action='store_true', help='从checkpoint继续（默认行为）')
    parser.add_argument('--limit', type=int, help='限制处理的文档数量（用于测试）')
    args = parser.parse_args()
    
    print("=" * 60)
    print("批量文档处理脚本")
    print("=" * 60)
    
    # 1. 清空数据（如果指定）
    if args.clean:
        print("\n[步骤1] 清空现有数据...")
        clean_database()
        clean_processed_data()
    
    # 2. 加载checkpoint
    checkpoint = load_checkpoint()
    processed_set = set(checkpoint["processed_files"])
    
    # 3. 获取待处理文档
    print("\n[步骤2] 获取文档列表...")
    all_provinces = get_province_docs()
    
    if args.province:
        provinces = {k: v for k, v in all_provinces.items() if k == args.province}
    else:
        provinces = all_provinces
    
    total_files = sum(len(files) for files in provinces.values())
    pending_files = []
    for prov, files in provinces.items():
        for f in files:
            key = f"{prov}:{f.name}"
            if key not in processed_set:
                pending_files.append((prov, f))
    
    print(f"总文档数: {total_files}")
    print(f"已处理: {len(processed_set)}")
    print(f"待处理: {len(pending_files)}")
    
    if args.limit:
        pending_files = pending_files[:args.limit]
        print(f"限制处理: {len(pending_files)}")
    
    # 4. 处理文档
    print("\n[步骤3] 开始处理...")
    
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(settings.database_url)
    Session = sessionmaker(bind=engine)
    
    stats = Counter()
    
    for i, (province, pdf_path) in enumerate(pending_files):
        key = f"{province}:{pdf_path.name}"
        print(f"\n[{i+1}/{len(pending_files)}] {province}/{pdf_path.name}")
        print(f"  大小: {pdf_path.stat().st_size / 1024:.1f} KB")
        
        success, msg = process_single_doc(pdf_path, province)
        
        if success:
            checkpoint["processed_files"].append(key)
            checkpoint["total_processed"] += 1
            stats["success"] += 1
            print(f"  {msg}")
            
            # 每10个文档保存checkpoint和入库
            if stats["success"] % 10 == 0:
                save_checkpoint(checkpoint)
                print(f"  [Checkpoint已保存]")
        else:
            checkpoint["failed_files"].append({"key": key, "error": msg})
            checkpoint["total_failed"] += 1
            stats["failed"] += 1
            print(f"  {msg}")
        
        checkpoint["last_province"] = province
        checkpoint["last_file"] = pdf_path.name
        
        # 每处理完一个文档就保存checkpoint（便于中断）
        save_checkpoint(checkpoint)
    
    # 5. 入库到数据库
    print("\n[步骤4] 入库到数据库...")
    
    with Session() as db:
        pipeline = IngestionPipeline(db=db, settings=settings)
        
        for province in provinces.keys():
            province_dir = PROCESSED_DIR / province
            if province_dir.exists():
                json_files = list(province_dir.glob("*.json"))
                if json_files:
                    print(f"\n入库 {province}: {len(json_files)} 个文档")
                    try:
                        result = pipeline.ingest_path(
                            path=province_dir,
                            province_code=province,
                            rebuild_index=False
                        )
                        db.commit()
                        print(f"  导入文档: {result['imported_documents']}")
                        print(f"  导入条款: {result['imported_clauses']}")
                        stats["imported_docs"] += result['imported_documents']
                        stats["imported_clauses"] += result['imported_clauses']
                    except Exception as e:
                        print(f"  入库失败: {e}")
    
    # 6. 最终统计
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"成功处理: {stats['success']}")
    print(f"失败文档: {stats['failed']}")
    print(f"入库文档: {stats['imported_docs']}")
    print(f"入库条款: {stats['imported_clauses']}")
    
    # 保存最终checkpoint
    checkpoint["completed_at"] = datetime.now().isoformat()
    checkpoint["final_stats"] = dict(stats)
    save_checkpoint(checkpoint)
    
    # 删除checkpoint（处理完成）
    Path(CHECKPOINT_FILE).unlink()
    print("\nCheckpoint已清除")


if __name__ == '__main__':
    main()
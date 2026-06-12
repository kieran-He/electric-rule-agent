"""
入库脚本 - 将 processed 目录的 JSON 文件导入数据库和向量库

用法：
python scripts/ingest_processed_docs.py [--province SX] [--rebuild-index]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.ingest.ingestion_pipeline import IngestionPipeline
from app.config import settings


PROCESSED_DIR = Path("data/processed")


def main():
    parser = argparse.ArgumentParser(description="入库 processed 文档")
    parser.add_argument("--province", type=str, help="只入库指定省份")
    parser.add_argument("--rebuild-index", action="store_true", help="重建向量索引")
    args = parser.parse_args()
    
    print("=" * 60)
    print("入库 processed 文档")
    print("=" * 60)
    print(f"向量库: {settings.chroma_path}")
    print(f"数据库: {settings.database_url}")
    
    # 获取待入库的省份目录
    if args.province:
        province_dirs = [PROCESSED_DIR / args.province]
    else:
        province_dirs = [d for d in PROCESSED_DIR.iterdir() if d.is_dir()]
    
    # 统计 JSON 文件数量
    total_json_files = 0
    for province_dir in province_dirs:
        if province_dir.exists():
            json_files = list(province_dir.glob("*.json"))
            total_json_files += len(json_files)
            print(f"  {province_dir.name}: {len(json_files)} 个文档")
    
    print(f"\n待入库文档总数: {total_json_files}")
    
    if total_json_files == 0:
        print("没有找到待入库的文档")
        return
    
    # 创建数据库连接
    engine = create_engine(settings.database_url)
    Session = sessionmaker(bind=engine)
    
    stats = {"imported_documents": 0, "imported_clauses": 0, "failed": 0}
    
    with Session() as db:
        pipeline = IngestionPipeline(db=db, settings=settings)
        
        for province_dir in province_dirs:
            if not province_dir.exists():
                continue
            
            json_files = list(province_dir.glob("*.json"))
            if not json_files:
                continue
            
            province_code = province_dir.name
            print(f"\n入库 {province_code}: {len(json_files)} 个文档...")
            
            try:
                result = pipeline.ingest_path(
                    path=province_dir,
                    province_code=province_code,
                    rebuild_index=args.rebuild_index,
                )
                db.commit()
                print(f"  导入文档: {result['imported_documents']}")
                print(f"  导入条款: {result['imported_clauses']}")
                stats["imported_documents"] += result["imported_documents"]
                stats["imported_clauses"] += result["imported_clauses"]
            except Exception as e:
                print(f"  入库失败: {e}")
                stats["failed"] += 1
    
    print("\n" + "=" * 60)
    print("入库完成")
    print("=" * 60)
    print(f"导入文档: {stats['imported_documents']}")
    print(f"导入条款: {stats['imported_clauses']}")
    print(f"失败: {stats['failed']}")


if __name__ == "__main__":
    main()
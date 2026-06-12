"""
同步更新数据库中的 issuer 和 issue_date 字段

用法：
python scripts/sync_metadata_to_db.py [--province SX]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

from sqlalchemy import create_engine, text


PROCESSED_DIR = Path("data/processed")


def main():
    parser = argparse.ArgumentParser(description="同步 JSON 元数据到数据库")
    parser.add_argument("--province", type=str, help="只同步指定省份")
    args = parser.parse_args()
    
    print("=" * 60)
    print("同步 issuer/issue_date 到数据库")
    print("=" * 60)
    
    engine = create_engine("sqlite:///./data/processed/app.db")
    
    # 获取 JSON 文件
    if args.province:
        province_dirs = [PROCESSED_DIR / args.province]
    else:
        province_dirs = [d for d in PROCESSED_DIR.iterdir() if d.is_dir()]
    
    stats = {"updated": 0, "skipped": 0, "failed": 0}
    
    with engine.connect() as conn:
        for province_dir in province_dirs:
            if not province_dir.exists():
                continue
            
            for json_file in province_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    meta = data.get("metadata", {})
                    file_hash = json_file.stem
                    issuer = meta.get("issuer")
                    issue_date = meta.get("issue_date")
                    
                    if issuer or issue_date:
                        # 更新 document 表
                        if issue_date:
                            conn.execute(
                                text("UPDATE document SET issuer = :issuer, issue_date = :issue_date WHERE file_hash = :hash"),
                                {"issuer": issuer, "issue_date": str(issue_date), "hash": file_hash}
                            )
                        else:
                            conn.execute(
                                text("UPDATE document SET issuer = :issuer WHERE file_hash = :hash"),
                                {"issuer": issuer, "hash": file_hash}
                            )
                        
                        stats["updated"] += 1
                        print(f"[{stats['updated']}] {json_file.name[:20]}... issuer={issuer}, date={issue_date}")
                except Exception as e:
                    stats["failed"] += 1
                    print(f"  失败: {json_file.name[:20]}... {e}")
        
        conn.commit()
    
    print("\n" + "=" * 60)
    print("同步完成")
    print("=" * 60)
    print(f"更新文档: {stats['updated']}")
    print(f"失败: {stats['failed']}")


if __name__ == "__main__":
    main()
"""
scripts/load_processed_to_db.py

简化版本（用户要求）：
- 文档哈希使用JSON标题（json_file.stem）
- 非必须字段以processed为准
- 仅导入ChromaDB（不导入SQL DB）
- 支持bge-large-zh embedding
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.config import settings
from app.repository import ChromaPolicyRepository


def map_clause_to_chroma_metadata(clause: dict, doc_title: str, idx: int) -> dict:
    """
    映射clause到ChromaDB metadata
    
    关键修改：
    - file_hash: 使用JSON文件名（json_file.stem）
    - 其他字段：直接传递processed值（非必须字段以processed为准）
    """
    return {
        "file_hash": doc_title,  # 使用JSON标题作为唯一标识
        "province_code": clause.get("province_code", "SN"),
        "source_name": clause.get("doc_name", ""),
        "doc_id": f"{doc_title}:{idx}",  # 唯一ID
        "doc_type": clause.get("doc_type", ""),
        "status": clause.get("doc_status", ""),
        "market_type": clause.get("doc_market_type", ""),
        "chapter_no": clause.get("chapter_no") or "",
        "section_no": clause.get("section_no") or "",
        "article_no": clause.get("article_no") or "",
        "title_path": clause.get("title_path", ""),
        "clause_text": clause.get("clause_text", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="导入processed JSON到ChromaDB")
    parser.add_argument("--path", default="data/processed", help="processed JSON目录路径")
    parser.add_argument("--rebuild", action="store_true", help="重建ChromaDB索引（清空旧数据）")
    args = parser.parse_args()

    print(f"Embedding Model: {settings.embedding_model}")
    print(f"ChromaDB Path: {settings.chroma_path}")

    processed_dir = Path(args.path)
    if not processed_dir.exists():
        print(f"错误: 目录不存在 {args.path}")
        sys.exit(1)

    repo = ChromaPolicyRepository(
        persist_directory=settings.chroma_path,
        embedding_model_name=settings.embedding_model,
    )

    if args.rebuild:
        try:
            repo._client.delete_collection("kb_sn")
            print("已清空旧ChromaDB索引")
        except Exception:
            pass

    json_files = list(processed_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith("_")]

    print(f"\n找到 {len(json_files)} 个processed JSON文件")

    stats = {"docs": 0, "clauses": 0, "failed": []}

    for json_file in json_files:
        doc_title = json_file.stem  # 使用JSON文件名作为标题和file_hash

        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            clauses = data.get("clauses", [])
            if not clauses:
                print(f"  {doc_title}: 无clauses，跳过")
                continue

            texts = []
            metadatas = []

            for idx, clause in enumerate(clauses):
                meta = map_clause_to_chroma_metadata(clause, doc_title, idx)
                texts.append(clause.get("clause_text", ""))
                metadatas.append(meta)

            count = repo.ingest_chunks(
                texts=texts,
                metadatas=metadatas,
                kb_scope="province",
                province_code="SN",
                rebuild=False,
            )

            stats["docs"] += 1
            stats["clauses"] += count
            print(f"  {doc_title}: {count} 条")

        except Exception as e:
            stats["failed"].append((json_file.name, str(e)))
            print(f"  {json_file.name}: 失败 - {e}")

    print(f"\n导入完成:")
    print(f"  成功: {stats['docs']} 文档, {stats['clauses']} 条款")
    if stats["failed"]:
        print(f"  失败: {len(stats['failed'])} 个")
        for name, err in stats["failed"]:
            print(f"    - {name}: {err}")

    print(f"\nChromaDB collection: kb_sn")
    print(f"Embedding model: {repo.embedder_name}")


if __name__ == "__main__":
    main()
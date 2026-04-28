"""
重建知识库：清空 MySQL + ChromaDB，从 JSON 导入
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.db.session import SessionLocal
from app.db.models.document import Document
from app.db.models.clause import Clause
from app.services.ingest.ingestion_pipeline import IngestionPipeline
import chromadb


def clear_mysql():
    """清空 MySQL Document/Clause 表"""
    with SessionLocal() as db:
        deleted_clauses = db.query(Clause).delete()
        deleted_docs = db.query(Document).delete()
        db.commit()
        print(f"MySQL: 已删除 {deleted_docs} 个文档, {deleted_clauses} 个条款")


def clear_chroma():
    """清空 ChromaDB kb_sn collection"""
    client = chromadb.PersistentClient(path=settings.chroma_path)
    try:
        client.delete_collection('kb_sn')
        print("ChromaDB: kb_sn collection 已删除")
    except Exception as e:
        print(f"ChromaDB: 删除失败（可能不存在）: {e}")


def import_from_json():
    """从 JSON 导入数据"""
    with SessionLocal() as db:
        pipeline = IngestionPipeline(db, settings)
        result = pipeline.ingest_path(
            Path("data/processed"),
            province_code="SN",
            rebuild_index=True
        )
        db.commit()
        print(f"导入完成: {result}")
        return result


def verify():
    """验证导入结果"""
    client = chromadb.PersistentClient(path=settings.chroma_path)
    try:
        collection = client.get_collection("kb_sn")
        count = collection.count()
        print(f"验证: ChromaDB kb_sn 包含 {count} 条文档")

        sample = collection.peek(limit=1)
        if sample['metadatas']:
            fields = list(sample['metadatas'][0].keys())
            print(f"Metadata 字段: {fields}")
            required = ['doc_name', 'page_start', 'page_end', 'title_path']
            missing = [f for f in required if f not in fields]
            if missing:
                print(f"警告: 缺少字段 {missing}")
            else:
                print("验证成功: 所有必需字段都存在")
    except Exception as e:
        print(f"验证失败: {e}")


def main():
    print("=" * 50)
    print("重建知识库")
    print("=" * 50)

    clear_mysql()
    clear_chroma()
    import_from_json()
    verify()

    print("=" * 50)
    print("重建完成")


if __name__ == "__main__":
    main()
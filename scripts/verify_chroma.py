"""
验证ChromaDB导入结果
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from app.config import settings
from app.repository import ChromaPolicyRepository

print(f"Embedding Model: {settings.embedding_model}")
print(f"Chroma Path: {settings.chroma_path}\n")

repo = ChromaPolicyRepository(
    persist_directory=settings.chroma_path,
    embedding_model_name=settings.embedding_model,
)

print(f"Repository ready: {repo.ready}")
print(f"Embedder name: {repo.embedder_name}\n")

if repo.ready:
    # 测试检索
    test_queries = [
        "陕西2026年中长期签约比例要求",
        "现货市场交易规则",
        "结算周期",
    ]
    
    for query in test_queries:
        print(f"查询: {query}")
        try:
            chunks = repo.retrieve(
                query=query,
                top_k=3,
                kb_scope="province",
                province_code="SN",
            )
            
            print(f"  检索到 {len(chunks)} 条结果:")
            for i, chunk in enumerate(chunks):
                print(f"    {i+1}. score={chunk.score:.3f}")
                print(f"       source: {chunk.metadata.get('source_name', 'N/A')[:50]}")
                print(f"       text: {chunk.text[:80]}...")
            print()
            
        except Exception as e:
            print(f"  错误: {e}\n")
    
    # 检查collection数量
    try:
        collection = repo._get_or_create_collection("kb_sn")
        count = collection.count()
        print(f"ChromaDB kb_sn collection 总数: {count} 条")
    except Exception as e:
        print(f"无法获取collection count: {e}")

else:
    print(f"Repository未就绪: {repo.init_error}")
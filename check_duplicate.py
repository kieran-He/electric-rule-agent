"""检查重复入库情况"""
import sqlite3
import json
from pathlib import Path

# 1. 检查manifest中的文件hash
manifest_path = Path("data/processed/SX/_manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

manifest_hashes = {d["file_hash"]: d["file_name"] for d in manifest.get("documents", [])}
failed_hashes = {d["file_hash"]: d["file_name"] for d in manifest.get("failed_documents", [])}

print("=" * 60)
print("Manifest状态:")
print(f"  成功处理: {len(manifest_hashes)} 个")
for h, name in manifest_hashes.items():
    print(f"    {h[:20]}... | {name}")
print(f"  处理失败: {len(failed_hashes)} 个")
for h, name in failed_hashes.items():
    print(f"    {h[:20]}... | {name}")

# 2. 检查SQLite
conn = sqlite3.connect("data/processed/app.db")
c = conn.cursor()

# 查看表结构
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]

print("\n" + "=" * 60)
print("SQLite数据库表:")
print(f"  表: {tables}")

sqlite_hashes = {}
sqlite_duplicates = []
clause_count = 0

if "document" in tables:
    c.execute("SELECT file_hash, doc_name FROM document WHERE province_code='SX'")
    sqlite_rows = c.fetchall()
    sqlite_hashes = {r[0]: r[1] for r in sqlite_rows}

    print(f"\n  SX文档数: {len(sqlite_hashes)} 个")
    for h, name in list(sqlite_hashes.items())[:5]:
        print(f"    {h[:20]}... | {name}")

    c.execute("""
        SELECT file_hash, COUNT(*) as cnt 
        FROM document 
        WHERE province_code='SX' 
        GROUP BY file_hash 
        HAVING cnt > 1
    """)
    sqlite_duplicates = c.fetchall()

    if sqlite_duplicates:
        print(f"\n  [WARN] SQLite重复文档: {len(sqlite_duplicates)} 个")
        for h, cnt in sqlite_duplicates:
            print(f"    {h[:20]}... | 重复{cnt}次")
    else:
        print(f"\n  [OK] SQLite无重复文档")

if "clause" in tables:
    c.execute("SELECT COUNT(*) FROM clause")
    clause_count = c.fetchone()[0]
    print(f"  总条款数: {clause_count} 条")
    
    # 通过doc_id关联查询SX的条款数
    c.execute("""
        SELECT COUNT(*) FROM clause c
        JOIN document d ON c.doc_id = d.id
        WHERE d.province_code = 'SX'
    """)
    sx_clause_count = c.fetchone()[0]
    print(f"  SX条款数: {sx_clause_count} 条")

conn.close()

# 3. 检查ChromaDB (直接查询collection)
try:
    import chromadb
    client = chromadb.PersistentClient(path="data/chroma")
    collections = client.list_collections()
    
    print("\n" + "=" * 60)
    print("ChromaDB collections:")
    print(f"  所有collections: {[c.name for c in collections]}")
    
    # 检查kb_sx
    sx_col = client.get_collection("kb_sx")
    sx_count = sx_col.count()
    print(f"\n  kb_sx文档数: {sx_count} 条")
    
    # 检查是否有重复的embedding_id
    sx_data = sx_col.get(limit=10)
    if sx_data and sx_data.get("ids"):
        print(f"  示例IDs: {sx_data['ids'][:3]}")
    
    chroma_hashes = {}
except Exception as e:
    print(f"\nChromaDB检查失败: {e}")
    chroma_hashes = {}
    chroma_hashes = {}

# 4. 对比分析
print("\n" + "=" * 60)
print("重复入库分析:")

# 新生成的文件hash
new_file_hash = "43fa5b97934b48bb9ae1ca9a580b3a1b5b5b232abfe226c9fdca4f5a0e97011c"
new_file_name = "山西电力市场规则体系（V16.0）.pdf"

if new_file_hash in manifest_hashes:
    print(f"  [OK] {new_file_name} 在manifest中已处理")
elif new_file_hash in failed_hashes:
    print(f"  [WARN] {new_file_name} 在manifest中标记为失败")
else:
    print(f"  [INFO] {new_file_name} 不在manifest中（刚用规则脚本生成）")

if new_file_hash in sqlite_hashes:
    print(f"  [WARN] {new_file_name} 已在SQLite中入库")
else:
    print(f"  [OK] {new_file_name} 未在SQLite中入库")

if new_file_hash in chroma_hashes:
    print(f"  [WARN] {new_file_name} 已在ChromaDB中入库")
else:
    print(f"  [OK] {new_file_name} 未在ChromaDB中入库")

print("=" * 60)
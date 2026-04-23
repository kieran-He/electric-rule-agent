#!/usr/bin/env python3
"""
Rebuild ChromaDB with COSINE distance metric
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ['EMBEDDING_MODEL'] = 'BAAI/bge-small-zh-v1.5'

import chromadb
from app.config import settings
from app.repository import ChromaPolicyRepository
import json

print('=== Rebuild ChromaDB with COSINE distance ===')

# Step 1: Reset ChromaDB
print('\n[Step 1] Resetting ChromaDB...')
client = chromadb.PersistentClient(path=settings.chroma_path)
cols = client.list_collections()
for c in cols:
    client.delete_collection(c.name)
    print(f'  Deleted: {c.name}')

# Step 2: Create collection with COSINE metric
print('\n[Step 2] Creating collection with COSINE metric...')
collection_sn = client.get_or_create_collection(
    name="kb_sn",
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)
print(f'  kb_sn metadata: {collection_sn.metadata}')

collection_global = client.get_or_create_collection(
    name="kb_global",
    metadata={"hnsw:space": "cosine"}
)
print(f'  kb_global metadata: {collection_global.metadata}')

# Step 3: Initialize repository
print('\n[Step 3] Creating repository...')
repo = ChromaPolicyRepository(settings.chroma_path, settings.embedding_model)
print(f'  Embedder: {repo.embedder_name}')
print(f'  Ready: {repo.ready}')

# Step 4: Import documents
print('\n[Step 4] Importing documents...')
processed_dir = Path('data/processed')
json_files = list(processed_dir.glob('*.json'))
json_files = [f for f in json_files if not f.name.startswith('_')]
print(f'  Files: {len(json_files)}')

total = 0
failed = []

def safe_meta(v):
    if v is None: return ""
    if isinstance(v, (str, int, float, bool)): return str(v) if not isinstance(v, bool) else v
    return str(v)

for i, json_file in enumerate(json_files):
    try:
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)
        
        clauses = data.get('clauses', [])
        if not clauses: continue
        
        texts = [c.get('clause_text', '') for c in clauses]
        metas = []
        for idx, c in enumerate(clauses):
            meta = {
                'doc_id': f'{json_file.stem}:{idx}',
                'file_hash': json_file.stem,
                'source_name': safe_meta(c.get('source_name', '')),
                'title_path': safe_meta(c.get('title_path', '')),
                'article_no': safe_meta(c.get('article_no', '')),
                'policy_level': safe_meta(c.get('policy_level', 'formal')),
            }
            metas.append({k: v for k, v in meta.items() if v})
        
        count = repo.ingest_chunks(texts, metas, 'province', 'SN', False)
        total += count
        print(f'  [{i+1}/{len(json_files)}] {json_file.stem}: {count} chunks')
    except Exception as e:
        failed.append((json_file.name, str(e)))
        print(f'  [{i+1}/{len(json_files)}] FAILED: {e}')

print(f'\n=== Summary ===')
print(f'  Total: {total}')
print(f'  Failed: {len(failed)}')

# Step 5: Verify
print('\n[Step 5] Verification...')
coll = client.get_collection('kb_sn')
print(f'  Count: {coll.count()}')
print(f'  Metadata: {coll.metadata}')

# Test cosine distance
sample = coll.get(limit=1, include=['embeddings'])
if len(sample['embeddings']) > 0:
    vec = sample['embeddings'][0]
    norm = sum(x*x for x in vec)**0.5
    print(f'  Embedding dim: {len(vec)}')
    print(f'  Embedding norm: {norm:.4f} (should be ~1.0 for normalized)')

print('\n=== COSINE distance configured ===')
#!/usr/bin/env python3
"""
Rebuild ChromaDB with current embedding model (512-dim BAAI/bge-small-zh-v1.5)
Fixed metadata conversion
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

print(f'Embedding model: {settings.embedding_model}')
print(f'ChromaDB path: {settings.chroma_path}')

# Step 1: Reset ChromaDB
print('\n=== Step 1: Resetting ChromaDB ===')
client = chromadb.PersistentClient(path=settings.chroma_path)
cols = client.list_collections()
for c in cols:
    client.delete_collection(c.name)
    print(f'  Deleted collection: {c.name}')
print(f'  Deleted {len(cols)} collections')

# Step 2: Create repository with new embedding model
print('\n=== Step 2: Creating repository ===')
repo = ChromaPolicyRepository(settings.chroma_path, settings.embedding_model)
print(f'  Repo embedder: {repo.embedder_name}')
print(f'  Repo ready: {repo.ready}')

# Test embedding dimension
if repo.ready:
    test_vec = repo._embed(["test"])
    print(f'  Embedding dimension: {len(test_vec[0])}')

# Step 3: Import processed documents
print('\n=== Step 3: Importing documents ===')
processed_dir = Path('data/processed')
json_files = list(processed_dir.glob('*.json'))
json_files = [f for f in json_files if not f.name.startswith('_')]
print(f'  Found {len(json_files)} JSON files')

total_clauses = 0
failed_files = []

def safe_meta(value):
    """Convert value to safe ChromaDB metadata format"""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value) if not isinstance(value, bool) else value
    return str(value)

for i, json_file in enumerate(json_files):
    print(f'  Processing file {i+1}/{len(json_files)}: {json_file.stem}')
    
    try:
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)
        
        clauses = data.get('clauses', [])
        if not clauses:
            print(f'    Skipping - no clauses')
            continue
        
        texts = [c.get('clause_text', '') for c in clauses]
        metas = []
        for idx, c in enumerate(clauses):
            meta = {
                'doc_id': f'{json_file.stem}:{idx}',
                'file_hash': json_file.stem,
                'source_name': safe_meta(c.get('source_name', json_file.stem)),
                'title_path': safe_meta(c.get('title_path', '')),
                'article_no': safe_meta(c.get('article_no', '')),
                'policy_level': safe_meta(c.get('policy_level', 'formal')),
                'page_start': safe_meta(c.get('page_start', '')),
                'page_end': safe_meta(c.get('page_end', '')),
            }
            # Remove None values
            meta = {k: v for k, v in meta.items() if v is not None and v != ""}
            metas.append(meta)
        
        print(f'    Ingesting {len(texts)} clauses...')
        count = repo.ingest_chunks(texts, metas, 'province', 'SN', False)
        total_clauses += count
        print(f'    Done: {count} clauses')
        
    except Exception as e:
        failed_files.append((json_file.name, str(e)))
        print(f'    FAILED: {e}')

print(f'\n=== Import Summary ===')
print(f'  Total clauses: {total_clauses}')
print(f'  Failed files: {len(failed_files)}')
for name, err in failed_files:
    print(f'    - {name}: {err}')

# Step 4: Verify
print('\n=== Step 4: Verification ===')
coll = client.get_or_create_collection('kb_sn')
print(f'  kb_sn count: {coll.count()}')
sample = coll.get(limit=1, include=['embeddings'])
if len(sample['embeddings']) > 0:
    print(f'  Sample embedding dim: {len(sample["embeddings"][0])}')

print('\n=== Rebuild Complete ===')
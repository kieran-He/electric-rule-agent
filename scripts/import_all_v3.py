import sys
from pathlib import Path
import os
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ['EMBEDDING_MODEL'] = 'BAAI/bge-large-zh'

from app.config import settings
from app.repository import ChromaPolicyRepository
import json

print(f'Embedding model: {settings.embedding_model}')

repo = ChromaPolicyRepository(settings.chroma_path, settings.embedding_model)
print(f'Repo embedder: {repo.embedder_name}')

processed_dir = Path('data/processed')
json_files = list(processed_dir.glob('*.json'))
json_files = [f for f in json_files if not f.name.startswith('_')]
print(f'Found {len(json_files)} JSON files')

total_clauses = 0
total_time = 0
failed_files = []

first_file = True

for i, json_file in enumerate(json_files):
    print(f'[{i+1}/{len(json_files)}] {json_file.stem}...', flush=True)
    
    try:
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)
        
        clauses = data.get('clauses', [])
        if not clauses:
            print(f'  skip (no clauses)')
            continue
        
        texts = [c.get('clause_text', '') for c in clauses]
        metas = []
        for idx, c in enumerate(clauses):
            meta = {
                'doc_id': f'{json_file.stem}:{idx}',
                'file_hash': json_file.stem,
                'source_name': str(c.get('doc_name', '') or '')[:200],
                'title_path': str(c.get('title_path', '') or '')[:200],
                'article_no': str(c.get('article_no', '') or ''),
            }
            metas.append(meta)
        
        start = time.time()
        rebuild = first_file
        count = repo.ingest_chunks(texts, metas, 'province', 'SN', rebuild=rebuild)
        first_file = False
        elapsed = time.time() - start
        total_time += elapsed
        total_clauses += count
        print(f'  {count} clauses in {elapsed:.1f}s', flush=True)
        
    except Exception as e:
        failed_files.append((json_file.name, str(e)))
        print(f'  FAILED: {e}', flush=True)

print(f'\nSummary:')
print(f'  Total: {total_clauses} clauses in {total_time:.1f}s')
print(f'  Failed: {len(failed_files)}')
for name, err in failed_files:
    print(f'    - {name}: {err}')
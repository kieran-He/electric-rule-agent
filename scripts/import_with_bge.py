import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ['EMBEDDING_MODEL'] = 'BAAI/bge-large-zh'

from app.config import settings
from app.repository import ChromaPolicyRepository
import json

print(f'Embedding model: {settings.embedding_model}')
print(f'ChromaDB path: {settings.chroma_path}')

repo = ChromaPolicyRepository(settings.chroma_path, settings.embedding_model)
print(f'Repo embedder: {repo.embedder_name}')
print(f'Repo ready: {repo.ready}')

processed_dir = Path('data/processed')
json_files = list(processed_dir.glob('*.json'))
json_files = [f for f in json_files if not f.name.startswith('_')]
print(f'Found {len(json_files)} JSON files')

total_clauses = 0
failed_files = []

for i, json_file in enumerate(json_files):
    print(f'Processing file {i+1}/{len(json_files)}: {json_file.stem}')
    
    try:
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)
        
        clauses = data.get('clauses', [])
        if not clauses:
            print(f'  Skipping - no clauses')
            continue
        
        texts = [c.get('clause_text', '') for c in clauses]
        metas = [{'doc_id': f'{json_file.stem}:{idx}', 'file_hash': json_file.stem} for idx in range(len(clauses))]
        
        print(f'  Ingesting {len(texts)} clauses...')
        count = repo.ingest_chunks(texts, metas, 'province', 'SN', False)
        total_clauses += count
        print(f'  Done: {count} clauses')
        
    except Exception as e:
        failed_files.append((json_file.name, str(e)))
        print(f'  FAILED: {e}')

print(f'\nImport complete:')
print(f'  Total clauses: {total_clauses}')
print(f'  Failed files: {len(failed_files)}')
for name, err in failed_files:
    print(f'    - {name}: {err}')
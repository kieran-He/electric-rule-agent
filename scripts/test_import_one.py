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
json_file = next(processed_dir.glob('*.json'))
print(f'Testing with: {json_file.stem}')

with open(json_file, encoding='utf-8') as f:
    data = json.load(f)

clauses = data.get('clauses', [])[:20]  # Just 20 clauses
texts = [c.get('clause_text', '') for c in clauses]
metas = [{'doc_id': f'{json_file.stem}:{idx}', 'file_hash': json_file.stem} for idx in range(len(clauses))]

print(f'Ingesting {len(texts)} clauses...')
start = time.time()
count = repo.ingest_chunks(texts, metas, 'province', 'SN', False)
elapsed = time.time() - start
print(f'Done: {count} clauses in {elapsed:.1f}s')

chunks = repo.retrieve('陕西中长期签约比例', top_k=3, kb_scope='province', province_code='SN')
print(f'Retrieved {len(chunks)} chunks')
for c in chunks:
    print(f'score={c.score:.3f}, doc_id={c.metadata.get("doc_id")}')
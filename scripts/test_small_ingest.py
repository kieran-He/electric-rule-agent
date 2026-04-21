import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.repository import ChromaPolicyRepository
import json

print(f'Embedding model: {settings.embedding_model}')

repo = ChromaPolicyRepository(settings.chroma_path, settings.embedding_model)
print(f'Repo embedder: {repo.embedder_name}')

processed_dir = Path('data/processed')
json_file = next(processed_dir.glob('*.json'))
print(f'Testing with file: {json_file.name}')

with open(json_file, encoding='utf-8') as f:
    data = json.load(f)

clauses = data.get('clauses', [])
texts = [c.get('clause_text', '') for c in clauses[:10]]
metas = [{'doc_id': str(i)} for i in range(len(texts))]
print(f'Testing with {len(texts)} texts')

result = repo.ingest_chunks(texts, metas, 'province', 'SN', False)
print(f'Ingested {result} chunks')
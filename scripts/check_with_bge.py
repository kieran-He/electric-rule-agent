import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ['EMBEDDING_MODEL'] = 'BAAI/bge-large-zh'

from app.config import settings
from app.repository import ChromaPolicyRepository

print(f'Embedding model: {settings.embedding_model}')

repo = ChromaPolicyRepository(settings.chroma_path, settings.embedding_model)
print(f'Repo embedder: {repo.embedder_name}')

chunks = repo.retrieve('陕西中长期签约比例', top_k=3, kb_scope='province', province_code='SN')
print(f'Found {len(chunks)} chunks')
for c in chunks:
    print(f'score={c.score:.3f}')
    print(f'  doc_id={c.metadata.get("doc_id", "N/A")}')
    print(f'  title_path={c.metadata.get("title_path", "N/A")}')
    print()
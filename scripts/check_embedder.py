import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.repository import ChromaPolicyRepository
from app.config import settings

print(f'Settings embedding model: {settings.embedding_model}')

repo = ChromaPolicyRepository(settings.chroma_path, settings.embedding_model)
print(f'Repo embedder name: {repo.embedder_name}')
print(f'Repo ready: {repo.ready}')
print(f'Repo init error: {repo.init_error}')

if repo._embedder_name == 'deterministic-fallback':
    print('Checking why SentenceTransformer failed...')
    try:
        from sentence_transformers import SentenceTransformer
        test_model = SentenceTransformer(settings.embedding_model)
        print('SentenceTransformer model loaded OK in direct test')
        print(f'Model max_seq_length: {test_model.max_seq_length}')
    except Exception as e:
        print(f'SentenceTransformer failed: {type(e).__name__}: {e}')
else:
    print('SentenceTransformer is being used')
    print(f'Embedder type: {type(repo._embedder)}')
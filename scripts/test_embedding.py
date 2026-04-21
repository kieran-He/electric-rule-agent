import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-large-zh')
    print('Model loaded successfully')
    result = model.encode(['test query'])
    print(f'Embedding dimension: {len(result[0])}')
except Exception as e:
    print(f'Error: {e}')
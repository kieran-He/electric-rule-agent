import chromadb
from pathlib import Path

chroma_path = './data/chroma'
client = chromadb.PersistentClient(path=chroma_path)

collections = client.list_collections()
print(f'Collections: {len(collections)}')

for c in collections:
    print(f'  Name: {c.name}')
    print(f'  ID: {c.id}')
    count = c.count()
    print(f'  Count: {count}')
    peek = c.peek(limit=1)
    embeddings = peek.get('embeddings')
    if embeddings is not None and len(embeddings) > 0:
        print(f'  Embedding dim: {len(embeddings[0])}')
    else:
        print(f'  Embedding dim: unknown')
    print()
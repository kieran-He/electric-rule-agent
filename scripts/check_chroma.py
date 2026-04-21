#!/usr/bin/env python3
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')
import chromadb
from app.config import settings

client = chromadb.PersistentClient(path=settings.chroma_path)
collections = client.list_collections()
for c in collections:
    print(f'Collection: {c.name}')
    print(f'Metadata: {c.metadata}')
    count = c.count()
    print(f'Count: {count}')
    if count > 0:
        sample = c.peek(limit=1)
        embeddings = sample.get("embeddings")
        if embeddings is not None and len(embeddings) > 0:
            print(f'Sample embedding dim: {len(embeddings[0])}')
        else:
            print(f'Sample embedding: N/A')
#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()
import chromadb
from app.config import settings

print(f'CHROMA_PATH from .env: {settings.chroma_path}')

client = chromadb.PersistentClient(path=settings.chroma_path)
collections = client.list_collections()
print(f'Collections: {[c.name for c in collections]}')
for c in collections:
    count = c.count()
    print(f'  {c.name}: {count} documents')
    if count > 0:
        sample = c.peek(limit=1)
        embeddings = sample.get('embeddings')
        if embeddings is not None and len(embeddings) > 0:
            dim = len(embeddings[0])
            print(f'    Embedding dim: {dim}')
#!/usr/bin/env python3
import chromadb

client = chromadb.PersistentClient(path='./data/chroma')
collections = client.list_collections()
print(f'Collections: {[c.name for c in collections]}')
for c in collections:
    print(f'{c.name}: {c.count()} documents')
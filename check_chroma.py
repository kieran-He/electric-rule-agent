import chromadb
from pathlib import Path

client = chromadb.PersistentClient(path='./data/chroma')
collections = client.list_collections()

print('=== ChromaDB Collections ===')
for c in collections:
    count = c.count()
    print(f'{c.name}: {count} documents')

print()
print('=== kb_sd Collection ===')
try:
    sd_collection = client.get_collection('kb_sd')
    print(f'kb_sd has {sd_collection.count()} documents')
    if sd_collection.count() > 0:
        sample = sd_collection.get(limit=3, include=['documents', 'metadatas'])
        for i, doc in enumerate(sample['documents'][:3]):
            meta = sample['metadatas'][i]
            print(f'Doc {i+1}: {doc[:80]}...')
            print(f'  Source: {meta.get("source_name", meta.get("doc_name", "N/A"))}')
except Exception as e:
    print(f'kb_sd error: {e}')

print()
print('=== kb_sn Collection ===')
try:
    sn_collection = client.get_collection('kb_sn')
    print(f'kb_sn has {sn_collection.count()} documents')
except Exception as e:
    print(f'kb_sn error: {e}')

print()
print('=== Supported Provinces ===')
supported = [c.name.replace('kb_', '').upper() for c in collections if c.name.startswith('kb_') and c.name != 'kb_global']
print(f'Provinces in ChromaDB: {supported}')
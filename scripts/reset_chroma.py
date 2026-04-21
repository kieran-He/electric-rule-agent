import chromadb
from pathlib import Path

chroma_path = './data/chroma'
client = chromadb.PersistentClient(path=chroma_path)
cols = client.list_collections()
for c in cols:
    client.delete_collection(c.name)
    print(f'Deleted collection: {c.name}')
print(f'Deleted {len(cols)} collections')
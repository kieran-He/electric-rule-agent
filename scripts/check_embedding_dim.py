import chromadb
client = chromadb.PersistentClient(path='./data/chroma')
coll = client.get_collection('kb_sn')
result = coll.get(limit=1, include=['embeddings'])
print(f'Embedding dim: {len(result["embeddings"][0])}')
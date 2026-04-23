import chromadb

client = chromadb.PersistentClient(path='./data/chroma')
coll = client.get_collection('kb_sn')

print(f"Collection name: {coll.name}")
print(f"Collection count: {coll.count()}")
print(f"Collection metadata: {coll.metadata}")

# Check if distance function is specified
if coll.metadata:
    print(f"\nDistance metric: {coll.metadata.get('hnsw:space', 'l2 (default)')}")
else:
    print("\nNo metadata found - using default L2 distance")

# Test query to see distance values
result = coll.get(limit=2, include=['embeddings', 'documents'])
if result['embeddings']:
    print(f"\nSample embedding dimension: {len(result['embeddings'][0])}")
    print(f"Sample embedding norm: {sum(x*x for x in result['embeddings'][0])**0.5:.4f}")
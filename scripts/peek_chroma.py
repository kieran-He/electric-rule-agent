import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from app.config import settings
import chromadb

client = chromadb.PersistentClient(path=settings.chroma_path)
collection = client.get_collection("kb_sn")

print(f"Collection: kb_sn")
print(f"Count: {collection.count()}")

if collection.count() > 0:
    sample = collection.peek(limit=3)
    print(f"\nSample documents:")
    for i, doc in enumerate(sample.get("documents", [])):
        print(f"  [{i}] {doc[:100]}...")
    
    print(f"\nSample metadatas:")
    for i, meta in enumerate(sample.get("metadatas", [])):
        print(f"  [{i}] file_hash={meta.get('file_hash', 'N/A')[:50]}")
        print(f"       source_name={meta.get('source_name', 'N/A')[:50]}")
        print(f"       title_path={meta.get('title_path', 'N/A')[:50]}")
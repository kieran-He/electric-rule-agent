"""Debug ChromaDB directly"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.repository import ChromaPolicyRepository
from app.config import settings

print(f"Chroma path: {settings.chroma_path}")
print(f"Embedding model: {settings.embedding_model}")

repo = ChromaPolicyRepository(settings.chroma_path, "deterministic")

try:
    col = repo._client.get_collection("kb_sn")
    print(f"kb_sn collection count: {col.count()}")
    
    if col.count() > 0:
        sample = col.peek(limit=3)
        print(f"Sample metadatas: {sample.get('metadatas', [])[:1]}")
        
        chunks = repo.retrieve("陕西中长期签约比例", top_k=3, kb_scope="province", province_code="SN")
        print(f"Retrieved {len(chunks)} chunks")
        for c in chunks:
            print(f"- score={c.score:.3f} source={c.metadata.get('source_name', '')[:30]}")
except Exception as e:
    print(f"Error: {e}")
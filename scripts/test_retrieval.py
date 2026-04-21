"""Test retrieval from ChromaDB"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.repository import ChromaPolicyRepository
from app.config import settings

print(f"Settings embedding_model: {settings.embedding_model}")

repo = ChromaPolicyRepository(settings.chroma_path, settings.embedding_model)
print(f"Repo embedder_name: {repo.embedder_name}")

try:
    chunks = repo.retrieve("陕西2026年中长期签约比例", top_k=3, kb_scope="province", province_code="SN")
    print(f"Retrieved {len(chunks)} chunks")
    for c in chunks:
        source = c.metadata.get("source_name", "")[:50]
        print(f"- score={c.score:.3f} | {source}...")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
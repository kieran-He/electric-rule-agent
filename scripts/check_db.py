#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.repository import ChromaPolicyRepository
from app.config import settings
from app.db.session import SessionLocal
from app.db.models.document import Document
from app.db.models.clause import Clause
from sqlalchemy import select, func

repo = ChromaPolicyRepository(settings.chroma_path, settings.embedding_model)
print(f'Embedding model: {repo.embedder_name}')
chunks = repo.retrieve('陕西中长期签约比例', top_k=3, kb_scope='province', province_code='SN')
print(f'Found {len(chunks)} chunks')
for c in chunks:
    print(f'score={c.score:.3f}')
    print(f'  file_hash={c.metadata.get("file_hash", "N/A")[:50]}')
    print(f'  doc_id={c.metadata.get("doc_id", "N/A")}')
    print(f'  source_name={c.metadata.get("source_name", "N/A")[:50]}')
    print(f'  title_path={c.metadata.get("title_path", "N/A")[:50]}')
    print()

with SessionLocal() as db:
    doc_count = db.scalar(select(func.count()).select_from(Document))
    clause_count = db.scalar(select(func.count()).select_from(Clause))
    print(f'SQL DB: {doc_count} documents, {clause_count} clauses')


def reset_chroma():
    import chromadb
    client = chromadb.PersistentClient(path=settings.chroma_path)
    cols = client.list_collections()
    for c in cols:
        client.delete_collection(c.name)
    print(f'Deleted {len(cols)} ChromaDB collections')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-chroma", action="store_true")
    args = parser.parse_args()
    if args.reset_chroma:
        reset_chroma()
#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.config import settings
from app.repository import ChromaPolicyRepository

repo = ChromaPolicyRepository(
    persist_directory=settings.chroma_path,
    embedding_model_name=settings.embedding_model,
)

chunks = repo.retrieve(
    query='陕西2026年中长期签约比例',
    top_k=5,
    kb_scope='province',
    province_code='SN',
)

print(f'Found {len(chunks)} chunks')
for i, c in enumerate(chunks[:3], 1):
    print()
    source = c.metadata.get("source_name", "")
    title = c.metadata.get("title_path", "")
    print(f'{i}. Source: {source[:60]}...')
    print(f'   Title: {title[:60]}...')
    print(f'   Score: {c.score:.3f}')
    print(f'   Text: {c.text[:400]}...')
    print()
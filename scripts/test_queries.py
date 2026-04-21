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

# Test different queries
queries = [
    '陕西2026年中长期签约比例',
    '发电侧中长期合同签约比例要求',
    '燃煤发电企业年度签约比例60%',
    '用电侧签约比例45%',
]

for query in queries:
    print(f'\\nQuery: {query}')
    print('='*60)
    chunks = repo.retrieve(
        query=query,
        top_k=3,
        kb_scope='province',
        province_code='SN',
    )
    for i, c in enumerate(chunks[:2], 1):
        source = c.metadata.get("source_name", "")
        title = c.metadata.get("title_path", "")
        print(f'{i}. Score: {c.score:.3f}')
        print(f'   Title: {title[:50]}...')
        print(f'   Text: {c.text[:150]}...')
        if '60%' in c.text or '45%' in c.text or '80%' in c.text:
            print('   [CONTAINS PERCENTAGE!]')
    print()
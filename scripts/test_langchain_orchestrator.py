#!/usr/bin/env python3
"""
Test LangChain QA Orchestrator directly (without API server)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from app.config import settings
from app.langchain.orchestrator import LangChainQAOrchestrator
from app.schemas.query import QueryRequest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

print(f"USE_LANGCHAIN: {settings.use_langchain}")
print(f"Chroma path: {settings.chroma_path}")
print(f"Embedding model: {settings.embedding_model}")

engine = create_engine(settings.database_url)
Session = sessionmaker(bind=engine)

req = QueryRequest(
    query="发电侧中长期合同签约比例要求是什么？",
    session_id="test_langchain",
    province_codes=["SN"],
    mode="province_only",
    top_k=3,
    need_citation=True,
)

print(f"\nQuery: {req.query}")
print("Running LangChainQAOrchestrator...")

import time
start = time.time()

with Session() as db:
    orchestrator = LangChainQAOrchestrator(db=db, settings=settings)
    result = orchestrator.run(req)

elapsed = time.time() - start

print(f"\nTime: {elapsed:.2f}s")
print(f"Answer: {result.answer[:300]}...")
print(f"Citations: {len(result.citations)}")
print(f"Intent: {result.intent}")
print(f"Confidence: {result.confidence}")
print(f"Warnings: {result.warnings}")
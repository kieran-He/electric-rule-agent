"""Test Query Rewrite + Hybrid Retrieval Chain."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from app.config import settings
from app.repository import ChromaPolicyRepository
from app.langchain.bm25_indexer import BM25Indexer
from app.langchain.hybrid_retriever import HybridRetriever, BGEReranker
from app.langchain.query_rewriter import QueryRewriter
from app.langchain.llm import MiniMaxLLMWrapper

print("=" * 60)
print("Query Rewrite + Hybrid Retrieval Chain Test")
print("=" * 60)

print(f"\nConfig: query_rewrite_enabled={settings.query_rewrite_enabled}")
print(f"Config: hybrid_vector_top_k={settings.hybrid_vector_top_k}")
print(f"Config: hybrid_final_top_k={settings.hybrid_final_top_k}")

print("\n[Step 1] Initialize LLM Wrapper...")
try:
    llm_wrapper = MiniMaxLLMWrapper()
    print("LLM Wrapper initialized")
except Exception as e:
    print(f"[ERROR] LLM init failed: {e}")
    sys.exit(1)

print("\n[Step 2] Initialize Query Rewriter...")
rewriter = QueryRewriter(
    llm_wrapper=llm_wrapper,
    enabled=settings.query_rewrite_enabled,
    min_length=settings.query_rewrite_min_length,
)
print(f"Rewriter stats: {rewriter.get_stats()}")

print("\n[Step 3] Initialize Repository...")
repo = ChromaPolicyRepository(
    persist_directory=settings.chroma_path,
    embedding_model_name=settings.embedding_model,
)
print(f"Repository: {repo.embedder_name}")

print("\n[Step 4] Initialize BM25 Indexer...")
bm25 = BM25Indexer(k1=settings.bm25_k1, b=settings.bm25_b)
doc_count = bm25.build_index()
print(f"BM25 indexed: {doc_count} docs")

print("\n[Step 5] Initialize Hybrid Retriever...")
reranker = BGEReranker(
    model_name=settings.reranker_model,
    max_length=settings.reranker_max_length,
)
hybrid = HybridRetriever(
    vector_repo=repo,
    bm25_indexer=bm25,
    reranker=reranker,
    query_rewriter=rewriter,
    vector_top_k=settings.hybrid_vector_top_k,
    bm25_top_k=settings.hybrid_bm25_top_k,
    final_top_k=settings.hybrid_final_top_k,
    use_query_rewrite=settings.query_rewrite_enabled,
    query_rewrite_keep_original=settings.query_rewrite_keep_original,
)
print(f"Retriever stats: {hybrid.get_stats()}")

print("\n[Step 6] Test Query Rewrite (口语化问题)...")
test_query = "交易规则是什么"
print(f"Input query: '{test_query}'")

should_rewrite, reason = rewriter.should_rewrite(test_query)
print(f"Should rewrite: {should_rewrite}, Reason: {reason}")

result = rewriter.rewrite(test_query)
print(f"Rewritten query: '{result.rewritten_query}'")
print(f"Triggered: {result.triggered}, Confidence: {result.confidence}")

print("\n[Step 7] Test Hybrid Retrieval...")
chunks = hybrid.retrieve(test_query, ["SN"])
print(f"Retrieved: {len(chunks)} chunks")

if chunks:
    print("\nTop 3 results:")
    for i, c in enumerate(chunks[:3]):
        source = c.metadata.get("source_name", "")[:40]
        print(f"  [{i+1}] score={c.score:.3f} | {source}...")
        print(f"       text: {c.text[:80]}...")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
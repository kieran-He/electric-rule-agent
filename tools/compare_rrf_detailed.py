"""
Detailed RRF vs Simple Comparison (Before Reranker)

Compare Stage1 RRF scores vs Simple concatenation before reranker.
This reveals the true difference in RRF fusion quality.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("USE_RRF_FUSION", "true")

from app.config import settings
from app.core.repository import ChromaPolicyRepository, PolicyChunk
from app.langchain.bm25_indexer import BM25Indexer
from app.langchain.hybrid_retriever import HybridRetriever


def test_stage1_rrf():
    """Test Stage1 RRF fusion (Vector + BM25) before reranker."""
    
    repo = ChromaPolicyRepository(
        persist_directory=settings.chroma_path,
        embedding_model_name=settings.embedding_model,
    )
    
    bm25_indexer = BM25Indexer(k1=settings.bm25_k1, b=settings.bm25_b)
    bm25_indexer.build_index()
    
    retriever_rrf = HybridRetriever(
        vector_repo=repo,
        bm25_indexer=bm25_indexer,
        reranker=None,
        query_expander=None,
        query_rewriter=None,
        llm_wrapper=None,
        vector_top_k=12,
        bm25_top_k=12,
        final_top_k=15,
        use_query_expansion=False,
        use_query_rewrite=False,
        use_rrf_fusion=True,
        rrf_k=60,
        rrf_stage1_top_k=15,
        rrf_stage2_top_k=20,
    )
    
    retriever_simple = HybridRetriever(
        vector_repo=repo,
        bm25_indexer=bm25_indexer,
        reranker=None,
        query_expander=None,
        query_rewriter=None,
        llm_wrapper=None,
        vector_top_k=12,
        bm25_top_k=12,
        final_top_k=15,
        use_query_expansion=False,
        use_query_rewrite=False,
        use_rrf_fusion=False,
    )
    
    query = "陕西中长期交易流程"
    province = "SN"
    
    print(f"\n=== Stage1 RRF vs Simple Concatenation ===")
    print(f"Query: {query}")
    print(f"Province: {province}")
    print(f"Vector top_k: 12, BM25 top_k: 12")
    
    chunks_rrf, _ = retriever_rrf.retrieve(query, [province])
    chunks_simple, _ = retriever_simple.retrieve(query, [province])
    
    print(f"\nRRF results ({len(chunks_rrf)} chunks):")
    for i, c in enumerate(chunks_rrf[:10], 1):
        source = c.metadata.get("source_name", "unknown")[:40]
        print(f"  {i}. score={c.score:.4f} | {source}")
    
    print(f"\nSimple results ({len(chunks_simple)} chunks):")
    for i, c in enumerate(chunks_simple[:10], 1):
        source = c.metadata.get("source_name", "unknown")[:40]
        print(f"  {i}. score={c.score:.4f} | {source}")
    
    hashes_rrf = [hash(c.text[:100]) for c in chunks_rrf[:10]]
    hashes_simple = [hash(c.text[:100]) for c in chunks_simple[:10]]
    
    overlap = len(set(hashes_rrf) & set(hashes_simple))
    print(f"\nOverlap ratio top10: {overlap}/10 = {overlap/10:.1%}")
    
    order_diff_count = 0
    for i in range(min(10, len(hashes_rrf), len(hashes_simple))):
        if hashes_rrf[i] != hashes_simple[i]:
            order_diff_count += 1
    print(f"Order differences in top10: {order_diff_count}")
    
    return {
        "query": query,
        "rrf_count": len(chunks_rrf),
        "simple_count": len(chunks_simple),
        "overlap_top10": overlap,
        "order_diff_count": order_diff_count,
        "rrf_scores": [round(c.score, 4) for c in chunks_rrf[:10]],
        "simple_scores": [round(c.score, 4) for c in chunks_simple[:10]],
    }


def test_stage2_rrf():
    """Test Stage2 RRF fusion (multi-query) before reranker."""
    
    repo = ChromaPolicyRepository(
        persist_directory=settings.chroma_path,
        embedding_model_name=settings.embedding_model,
    )
    
    bm25_indexer = BM25Indexer(k1=settings.bm25_k1, b=settings.bm25_b)
    bm25_indexer.build_index()
    
    retriever_rrf = HybridRetriever(
        vector_repo=repo,
        bm25_indexer=bm25_indexer,
        reranker=None,
        query_expander=None,
        query_rewriter=None,
        llm_wrapper=None,
        vector_top_k=12,
        bm25_top_k=12,
        final_top_k=20,
        use_query_expansion=True,
        query_expansion_method="synonyms",
        query_expansion_max=2,
        use_query_rewrite=False,
        use_rrf_fusion=True,
        rrf_k=60,
        rrf_stage1_top_k=15,
        rrf_stage2_top_k=20,
    )
    
    retriever_simple = HybridRetriever(
        vector_repo=repo,
        bm25_indexer=bm25_indexer,
        reranker=None,
        query_expander=None,
        query_rewriter=None,
        llm_wrapper=None,
        vector_top_k=12,
        bm25_top_k=12,
        final_top_k=20,
        use_query_expansion=True,
        query_expansion_method="synonyms",
        query_expansion_max=2,
        use_query_rewrite=False,
        use_rrf_fusion=False,
    )
    
    query = "陕西中长期交易流程"
    province = "SN"
    
    print(f"\n\n=== Stage2 RRF (Multi-Query) vs Simple ===")
    print(f"Query: {query}")
    print(f"Province: {province}")
    print(f"Query expansion: synonyms (max 2)")
    
    chunks_rrf, _ = retriever_rrf.retrieve(query, [province])
    chunks_simple, _ = retriever_simple.retrieve(query, [province])
    
    print(f"\nRRF Stage2 results ({len(chunks_rrf)} chunks):")
    for i, c in enumerate(chunks_rrf[:10], 1):
        source = c.metadata.get("source_name", "unknown")[:40]
        print(f"  {i}. score={c.score:.4f} | {source}")
    
    print(f"\nSimple multi-query results ({len(chunks_simple)} chunks):")
    for i, c in enumerate(chunks_simple[:10], 1):
        source = c.metadata.get("source_name", "unknown")[:40]
        print(f"  {i}. score={c.score:.4f} | {source}")
    
    hashes_rrf = [hash(c.text[:100]) for c in chunks_rrf[:10]]
    hashes_simple = [hash(c.text[:100]) for c in chunks_simple[:10]]
    
    overlap = len(set(hashes_rrf) & set(hashes_simple))
    print(f"\nOverlap ratio top10: {overlap}/10 = {overlap/10:.1%}")
    
    order_diff_count = 0
    for i in range(min(10, len(hashes_rrf), len(hashes_simple))):
        if hashes_rrf[i] != hashes_simple[i]:
            order_diff_count += 1
    print(f"Order differences in top10: {order_diff_count}")
    
    return {
        "query": query,
        "rrf_count": len(chunks_rrf),
        "simple_count": len(chunks_simple),
        "overlap_top10": overlap,
        "order_diff_count": order_diff_count,
        "rrf_scores": [round(c.score, 4) for c in chunks_rrf[:10]],
        "simple_scores": [round(c.score, 4) for c in chunks_simple[:10]],
    }


def main():
    print("Testing RRF fusion quality...")
    
    stage1_result = test_stage1_rrf()
    stage2_result = test_stage2_rrf()
    
    output = {
        "stage1_single_query": stage1_result,
        "stage2_multi_query": stage2_result,
    }
    
    output_path = Path("data/rrf_detailed_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
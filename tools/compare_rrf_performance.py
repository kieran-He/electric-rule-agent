"""
RRF vs Simple Concatenation Performance Comparison

Compare retrieval quality between:
1. RRF fusion (use_rrf_fusion=True)
2. Simple concatenation (use_rrf_fusion=False)

Metrics:
- Number of unique documents retrieved
- Score distribution
- Top-K overlap ratio
- Retrieval latency
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("USE_RRF_FUSION", "true")

from app.config import settings
from app.core.repository import ChromaPolicyRepository, PolicyChunk
from app.langchain.bm25_indexer import BM25Indexer
from app.langchain.hybrid_retriever import HybridRetriever, BGEReranker
from app.langchain.query_expander import QueryExpander
from app.langchain.query_rewriter import QueryRewriter
from app.langchain.llm import MiniMaxLLMWrapper

TEST_QUERIES = [
    "陕西中长期交易流程",
    "电力市场准入条件",
]


def create_retriever(use_rrf: bool) -> HybridRetriever:
    """Create HybridRetriever with specified RRF setting."""
    repo = ChromaPolicyRepository(
        persist_directory=settings.chroma_path,
        embedding_model_name=settings.embedding_model,
    )
    
    bm25_indexer = BM25Indexer(
        k1=settings.bm25_k1,
        b=settings.bm25_b,
    )
    bm25_indexer.build_index()
    
    reranker = BGEReranker(
        model_name=settings.reranker_model,
        max_length=settings.reranker_max_length,
    )
    
    llm_wrapper = MiniMaxLLMWrapper(
        api_key=os.getenv("LLM_API_KEY", ""),
        endpoint=os.getenv("LLM_ENDPOINT", "https://api.minimaxi.com/anthropic"),
        model=os.getenv("LLM_MODEL", "MiniMax-M2.7"),
        disable_thinking=True,
    )
    
    query_expander = None
    if settings.query_expansion:
        query_expander = QueryExpander(
            max_expansions=settings.query_expansion_max,
            llm_wrapper=llm_wrapper,
        )
    
    query_rewriter = None
    if settings.query_rewrite_enabled:
        query_rewriter = QueryRewriter(
            llm_wrapper=llm_wrapper,
            enabled=True,
            always_rewrite=settings.query_rewrite_always,
        )
    
    retriever = HybridRetriever(
        vector_repo=repo,
        bm25_indexer=bm25_indexer,
        reranker=reranker,
        query_expander=None,
        query_rewriter=None,
        llm_wrapper=None,
        vector_top_k=settings.hybrid_vector_top_k,
        bm25_top_k=settings.hybrid_bm25_top_k,
        final_top_k=settings.hybrid_final_top_k,
        use_query_expansion=False,
        query_expansion_method=settings.query_expansion_method,
        query_expansion_max=settings.query_expansion_max,
        use_query_rewrite=False,
        query_rewrite_keep_original=settings.query_rewrite_keep_original,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        cache_dir="data/cache",
        use_rrf_fusion=use_rrf,
        rrf_k=settings.rrf_k,
        rrf_stage1_top_k=settings.rrf_stage1_top_k,
        rrf_stage2_top_k=settings.rrf_stage2_top_k,
    )
    
    return retriever


def calculate_overlap(chunks1: List[PolicyChunk], chunks2: List[PolicyChunk], top_k: int = 5) -> float:
    """Calculate overlap ratio between two chunk lists."""
    if not chunks1 or not chunks2:
        return 0.0
    
    texts1 = set(hash(c.text[:100]) for c in chunks1[:top_k])
    texts2 = set(hash(c.text[:100]) for c in chunks2[:top_k])
    
    overlap = len(texts1 & texts2)
    return overlap / min(len(texts1), len(texts2), top_k)


def get_chunk_ids(chunks: List[PolicyChunk]) -> List[int]:
    """Get chunk identifiers (hash of first 100 chars)."""
    return [hash(c.text[:100]) for c in chunks]


def compare_query(
    query: str,
    retriever_rrf: HybridRetriever,
    retriever_simple: HybridRetriever,
    province_codes: List[str] = ["SN"],
) -> Dict:
    """Compare retrieval results for a single query."""
    
    start_rrf = time.time()
    chunks_rrf, detected_rrf = retriever_rrf.retrieve(query, province_codes)
    latency_rrf = time.time() - start_rrf
    
    start_simple = time.time()
    chunks_simple, detected_simple = retriever_simple.retrieve(query, province_codes)
    latency_simple = time.time() - start_simple
    
    overlap_top5 = calculate_overlap(chunks_rrf, chunks_simple, top_k=5)
    overlap_top10 = calculate_overlap(chunks_rrf, chunks_simple, top_k=10)
    
    scores_rrf = [c.score for c in chunks_rrf[:10]]
    scores_simple = [c.score for c in chunks_simple[:10]]
    
    result = {
        "query": query,
        "detected_provinces": detected_rrf,
        "rrf_results_count": len(chunks_rrf),
        "simple_results_count": len(chunks_simple),
        "rrf_latency_ms": int(latency_rrf * 1000),
        "simple_latency_ms": int(latency_simple * 1000),
        "overlap_top5_ratio": round(overlap_top5, 3),
        "overlap_top10_ratio": round(overlap_top10, 3),
        "rrf_top5_scores": [round(s, 4) for s in scores_rrf[:5]],
        "simple_top5_scores": [round(s, 4) for s in scores_simple[:5]],
        "rrf_top5_sources": [c.metadata.get("source_name", "unknown")[:50] for c in chunks_rrf[:5]],
        "simple_top5_sources": [c.metadata.get("source_name", "unknown")[:50] for c in chunks_simple[:5]],
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare RRF vs Simple concatenation performance")
    parser.add_argument("--queries", nargs="+", default=TEST_QUERIES, help="Test queries")
    parser.add_argument("--province", default="SN", help="Province code")
    parser.add_argument("--output", default="data/rrf_comparison_results.json", help="Output JSON file")
    args = parser.parse_args()
    
    province_codes = [args.province.upper()]
    
    print("Initializing retrievers...")
    print(f"  RRF: use_rrf=True, rrf_k={settings.rrf_k}, stage1_top_k={settings.rrf_stage1_top_k}, stage2_top_k={settings.rrf_stage2_top_k}")
    print(f"  Simple: use_rrf=False (fallback to _merge_and_deduplicate)")
    
    retriever_rrf = create_retriever(use_rrf=True)
    retriever_simple = create_retriever(use_rrf=False)
    
    print(f"\nComparing {len(args.queries)} queries...")
    
    results = []
    for i, query in enumerate(args.queries, 1):
        print(f"\n[{i}/{len(args.queries)}] Query: {query}")
        try:
            result = compare_query(query, retriever_rrf, retriever_simple, province_codes)
            results.append(result)
            
            print(f"  RRF results: {result['rrf_results_count']} chunks, latency: {result['rrf_latency_ms']}ms")
            print(f"  Simple results: {result['simple_results_count']} chunks, latency: {result['simple_latency_ms']}ms")
            print(f"  Overlap top5: {result['overlap_top5_ratio']:.1%}, top10: {result['overlap_top10_ratio']:.1%}")
            print(f"  RRF scores: {result['rrf_top5_scores'][:3]}")
            print(f"  Simple scores: {result['simple_top5_scores'][:3]}")
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"query": query, "error": str(e)})
    
    summary = {
        "config": {
            "rrf_k": settings.rrf_k,
            "rrf_stage1_top_k": settings.rrf_stage1_top_k,
            "rrf_stage2_top_k": settings.rrf_stage2_top_k,
            "vector_top_k": settings.hybrid_vector_top_k,
            "bm25_top_k": settings.hybrid_bm25_top_k,
            "final_top_k": settings.hybrid_final_top_k,
            "province": args.province,
        },
        "results": results,
        "summary_stats": {
            "avg_rrf_latency_ms": sum(r.get("rrf_latency_ms", 0) for r in results if "rrf_latency_ms" in r) / len(results),
            "avg_simple_latency_ms": sum(r.get("simple_latency_ms", 0) for r in results if "simple_latency_ms" in r) / len(results),
            "avg_overlap_top5": sum(r.get("overlap_top5_ratio", 0) for r in results if "overlap_top5_ratio" in r) / len(results),
            "avg_overlap_top10": sum(r.get("overlap_top10_ratio", 0) for r in results if "overlap_top10_ratio" in r) / len(results),
        },
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n=== Summary ===")
    print(f"Average RRF latency: {summary['summary_stats']['avg_rrf_latency_ms']:.1f}ms")
    print(f"Average Simple latency: {summary['summary_stats']['avg_simple_latency_ms']:.1f}ms")
    print(f"Average overlap top5: {summary['summary_stats']['avg_overlap_top5']:.1%}")
    print(f"Average overlap top10: {summary['summary_stats']['avg_overlap_top10']:.1%}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
"""
Query Rewrite Retrieval Comparison

Compares retrieval performance with and without query rewrite.
Directly uses HybridRetriever for accurate comparison.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import List, Tuple
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.repository import ChromaPolicyRepository, PolicyChunk
from app.langchain.bm25_indexer import BM25Indexer
from app.langchain.hybrid_retriever import HybridRetriever, BGEReranker
from app.langchain.query_rewriter import QueryRewriter
from app.langchain.llm import MiniMaxLLMWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_benchmark_sample(benchmark_path: str, sample_size: int = 10) -> List[dict]:
    """Load and sample queries from benchmark."""
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = data.get("questions", [])
    if len(questions) <= sample_size:
        return questions
    
    random.seed(42)
    return random.sample(questions, sample_size)


def init_retriever_with_rewrite() -> HybridRetriever:
    """Initialize HybridRetriever with query rewrite enabled."""
    repo = ChromaPolicyRepository(
        persist_directory=settings.chroma_path,
        embedding_model_name=settings.embedding_model,
    )
    
    bm25_indexer = BM25Indexer(k1=settings.bm25_k1, b=settings.bm25_b)
    bm25_indexer.build_index()
    
    reranker = BGEReranker(
        model_name=settings.reranker_model,
        max_length=settings.reranker_max_length,
    )
    
    llm_wrapper = MiniMaxLLMWrapper()
    query_rewriter = QueryRewriter(
        llm_wrapper=llm_wrapper,
        enabled=True,
        min_length=settings.query_rewrite_min_length,
    )
    
    retriever = HybridRetriever(
        vector_repo=repo,
        bm25_indexer=bm25_indexer,
        reranker=reranker,
        query_rewriter=query_rewriter,
        vector_top_k=settings.hybrid_vector_top_k,
        bm25_top_k=settings.hybrid_bm25_top_k,
        final_top_k=settings.hybrid_final_top_k,
        use_query_rewrite=True,
        query_rewrite_keep_original=True,
    )
    
    return retriever


def init_retriever_no_rewrite() -> HybridRetriever:
    """Initialize HybridRetriever without query rewrite."""
    repo = ChromaPolicyRepository(
        persist_directory=settings.chroma_path,
        embedding_model_name=settings.embedding_model,
    )
    
    bm25_indexer = BM25Indexer(k1=settings.bm25_k1, b=settings.bm25_b)
    bm25_indexer.build_index()
    
    reranker = BGEReranker(
        model_name=settings.reranker_model,
        max_length=settings.reranker_max_length,
    )
    
    retriever = HybridRetriever(
        vector_repo=repo,
        bm25_indexer=bm25_indexer,
        reranker=reranker,
        query_rewriter=None,
        vector_top_k=settings.hybrid_vector_top_k,
        bm25_top_k=settings.hybrid_bm25_top_k,
        final_top_k=settings.hybrid_final_top_k,
        use_query_rewrite=False,
    )
    
    return retriever


def evaluate_retrieval(
    retriever: HybridRetriever,
    queries: List[dict],
    province_codes: List[str] = ["SN"],
) -> Tuple[float, float, List[dict]]:
    """
    Evaluate retrieval performance.
    
    Returns:
        avg_score: Average reranker score
        avg_doc_count: Average number of retrieved docs
        results: Detailed results for each query
    """
    results = []
    total_score = 0.0
    total_doc_count = 0
    
    for query_item in queries:
        question = query_item["question"]
        question_id = query_item["question_id"]
        expected_keywords = query_item.get("expected_answer_keywords", [])
        
        chunks = retriever.retrieve(question, province_codes)
        
        if chunks:
            avg_chunk_score = sum(c.score or 0.0 for c in chunks) / len(chunks)
            top_score = chunks[0].score or 0.0
        else:
            avg_chunk_score = 0.0
            top_score = 0.0
        
        keywords_hit_count = 0
        for chunk in chunks[:3]:
            chunk_text = chunk.text.lower()
            for kw in expected_keywords:
                if kw.lower() in chunk_text:
                    keywords_hit_count += 1
                    break
        
        keyword_recall = keywords_hit_count / max(len(expected_keywords), 1) if expected_keywords else 0.0
        
        result = {
            "question_id": question_id,
            "question": question,
            "doc_count": len(chunks),
            "top_score": top_score,
            "avg_score": avg_chunk_score,
            "keyword_recall": keyword_recall,
            "top_docs": [c.metadata.get("doc_name", c.metadata.get("doc_id", "")) for c in chunks[:3]],
            "rewritten_query": getattr(retriever, "_last_rewritten_query", None),
        }
        
        results.append(result)
        total_score += top_score
        total_doc_count += len(chunks)
    
    avg_score = total_score / len(queries) if queries else 0.0
    avg_doc_count = total_doc_count / len(queries) if queries else 0.0
    
    return avg_score, avg_doc_count, results


def compare_results(
    results_with_rewrite: List[dict],
    results_no_rewrite: List[dict],
) -> dict:
    """Compare retrieval results."""
    comparison = {
        "queries": [],
        "summary": {},
    }
    
    for r_with, r_no in zip(results_with_rewrite, results_no_rewrite):
        query_comparison = {
            "question_id": r_with["question_id"],
            "question": r_with["question"],
            "with_rewrite": {
                "doc_count": r_with["doc_count"],
                "top_score": r_with["top_score"],
                "keyword_recall": r_with["keyword_recall"],
                "top_docs": r_with["top_docs"],
            },
            "no_rewrite": {
                "doc_count": r_no["doc_count"],
                "top_score": r_no["top_score"],
                "keyword_recall": r_no["keyword_recall"],
                "top_docs": r_no["top_docs"],
            },
            "improvement": {
                "doc_count_delta": r_with["doc_count"] - r_no["doc_count"],
                "score_delta": r_with["top_score"] - r_no["top_score"],
                "recall_delta": r_with["keyword_recall"] - r_no["keyword_recall"],
                "better": r_with["keyword_recall"] > r_no["keyword_recall"],
            },
        }
        comparison["queries"].append(query_comparison)
    
    with_rewrite_scores = [r["top_score"] for r in results_with_rewrite]
    no_rewrite_scores = [r["top_score"] for r in results_no_rewrite]
    
    with_rewrite_recalls = [r["keyword_recall"] for r in results_with_rewrite]
    no_rewrite_recalls = [r["keyword_recall"] for r in results_no_rewrite]
    
    comparison["summary"] = {
        "avg_score_with_rewrite": sum(with_rewrite_scores) / len(with_rewrite_scores),
        "avg_score_no_rewrite": sum(no_rewrite_scores) / len(no_rewrite_scores),
        "avg_recall_with_rewrite": sum(with_rewrite_recalls) / len(with_rewrite_recalls),
        "avg_recall_no_rewrite": sum(no_rewrite_recalls) / len(no_rewrite_recalls),
        "improved_queries": sum(1 for q in comparison["queries"] if q["improvement"]["better"]),
        "degraded_queries": sum(1 for q in comparison["queries"] if not q["improvement"]["better"] and q["improvement"]["recall_delta"] < 0),
    }
    
    return comparison


def print_comparison_report(comparison: dict) -> None:
    """Print comparison report to console."""
    print("\n" + "=" * 80)
    print("Query Rewrite Retrieval Comparison Report")
    print("=" * 80)
    
    summary = comparison["summary"]
    print("\n【Summary】")
    print(f"  Avg Top Score (with rewrite):    {summary['avg_score_with_rewrite']:.4f}")
    print(f"  Avg Top Score (no rewrite):      {summary['avg_score_no_rewrite']:.4f}")
    print(f"  Score Improvement:               {summary['avg_score_with_rewrite'] - summary['avg_score_no_rewrite']:+.4f}")
    print()
    print(f"  Avg Keyword Recall (with rewrite): {summary['avg_recall_with_rewrite']:.4f}")
    print(f"  Avg Keyword Recall (no rewrite):   {summary['avg_recall_no_rewrite']:.4f}")
    print(f"  Recall Improvement:                {summary['avg_recall_with_rewrite'] - summary['avg_recall_no_rewrite']:+.4f}")
    print()
    print(f"  Improved Queries:   {summary['improved_queries']}")
    print(f"  Degraded Queries:   {summary['degraded_queries']}")
    
    print("\n【Detailed Results】")
    for q in comparison["queries"]:
        print(f"\n  Q{q['question_id']}: {q['question']}")
        print(f"    With Rewrite: docs={q['with_rewrite']['doc_count']}, score={q['with_rewrite']['top_score']:.3f}, recall={q['with_rewrite']['keyword_recall']:.2f}")
        print(f"    No Rewrite:   docs={q['no_rewrite']['doc_count']}, score={q['no_rewrite']['top_score']:.3f}, recall={q['no_rewrite']['keyword_recall']:.2f}")
        
        if q["improvement"]["better"]:
            print(f"    [+] Improved (recall +{q['improvement']['recall_delta']:.2f})")
        elif q["improvement"]["recall_delta"] < 0:
            print(f"    [-] Degraded (recall {q['improvement']['recall_delta']:.2f})")
        else:
            print(f"    [=] No change")
    
    print("\n" + "=" * 80)


def main():
    benchmark_path = "evaluation/benchmark.json"
    sample_size = 10
    output_path = "evaluation/reports/query_rewrite_comparison.json"
    
    print(f"Loading benchmark sample from {benchmark_path}")
    queries = load_benchmark_sample(benchmark_path, sample_size)
    print(f"Sample size: {len(queries)} queries")
    
    print("\nInitializing retriever with query rewrite...")
    retriever_with_rewrite = init_retriever_with_rewrite()
    
    print("\nInitializing retriever without query rewrite...")
    retriever_no_rewrite = init_retriever_no_rewrite()
    
    print("\nRunning retrieval evaluation with query rewrite...")
    avg_score_with, avg_docs_with, results_with = evaluate_retrieval(
        retriever_with_rewrite, queries
    )
    
    print("\nRunning retrieval evaluation without query rewrite...")
    avg_score_no, avg_docs_no, results_no = evaluate_retrieval(
        retriever_no_rewrite, queries
    )
    
    print("\nComparing results...")
    comparison = compare_results(results_with, results_no)
    
    print_comparison_report(comparison)
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print(f"\nComparison saved to {output_path}")


if __name__ == "__main__":
    main()
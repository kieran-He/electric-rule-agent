#!/usr/bin/env python3
"""
BM25 Hybrid Retrieval Experiment v3 - Optimized for Speed

Baseline: Vector-only (no BM25, no Rerank)
Hybrid-v3: Vector + BM25(k1=1.5,b=0.6) + bge-reranker-base (preloaded)

Optimizations:
- Use bge-reranker-base (201M params, faster than large)
- Disable verbose output
- Batch processing
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import os
import warnings
warnings.filterwarnings('ignore')

# Disable progress bars
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"

from app.config import settings
from app.repository import ChromaPolicyRepository, PolicyChunk
from app.langchain.bm25_indexer import BM25Indexer
from app.langchain.hybrid_retriever import HybridRetriever, BGEReranker
from app.langchain.reranker_cache import preload_reranker

print("=" * 70)
print("BM25 HYBRID RETRIEVAL EXPERIMENT V3 (Optimized)")
print("=" * 70)


@dataclass
class RetrievalResult:
    question_id: str
    question: str
    category: str
    baseline_time: float
    hybrid_time: float
    baseline_chunks: int
    hybrid_chunks: int
    baseline_avg_score: float
    hybrid_avg_score: float


def load_test_set() -> List[Dict]:
    """Load test questions."""
    with open("evaluation/benchmark_test.json", encoding='utf-8') as f:
        data = json.load(f)
    return data.get("questions", [])


def calculate_avg_score(chunks: List[PolicyChunk]) -> float:
    """Calculate average score."""
    if not chunks:
        return 0.0
    scores = [c.score for c in chunks if hasattr(c, 'score') and c.score > 0]
    return sum(scores) / len(scores) if scores else 0.0


def run_experiment():
    """Run experiment."""
    print("\nINITIALIZATION...")
    
    # Load test questions
    test_questions = load_test_set()
    print(f"Test questions: {len(test_questions)}")
    
    # Initialize Vector repo (silent)
    repo = ChromaPolicyRepository(
        persist_directory=settings.chroma_path,
        embedding_model_name=settings.embedding_model,
    )
    print(f"Vector repo: ready={repo.ready}")
    
    # Initialize BM25
    bm25_indexer = BM25Indexer(k1=1.5, b=0.6)
    bm25_docs = bm25_indexer.build_index()
    print(f"BM25 index: {bm25_docs} docs (k1=1.5, b=0.6)")
    
    # Preload Reranker (base model for speed)
    print("Preloading reranker (bge-reranker-base)...")
    preload_start = time.time()
    preload_reranker("BAAI/bge-reranker-base")
    preload_time = time.time() - preload_start
    print(f"Reranker preload: {preload_time:.2f}s")
    
    # Create Reranker instance
    reranker = BGEReranker(model_name="BAAI/bge-reranker-base")
    
    # Create Hybrid retriever
    hybrid_retriever = HybridRetriever(
        vector_repo=repo,
        bm25_indexer=bm25_indexer,
        reranker=reranker,
        vector_top_k=15,
        bm25_top_k=15,
        final_top_k=12,
        use_query_expansion=False,
    )
    
    results: List[RetrievalResult] = []
    
    print("\n" + "-" * 70)
    print("RUNNING EXPERIMENT (20 questions)")
    print("-" * 70)
    
    for i, q in enumerate(test_questions):
        question_id = q.get("question_id", f"q{i}")
        question_text = q.get("question", "")
        category = q.get("category", "unknown")
        
        # Progress indicator
        print(f"[{i+1}/{len(test_questions)}] {question_id}", end=" ", flush=True)
        
        # Baseline: Vector-only
        start = time.time()
        baseline_chunks = repo.retrieve(
            query=question_text,
            top_k=12,
            kb_scope="province",
            province_code="SN",
        )
        baseline_time = time.time() - start
        baseline_avg_score = calculate_avg_score(baseline_chunks)
        
        # Hybrid-v3: Optimized
        start = time.time()
        hybrid_chunks = hybrid_retriever.retrieve(question_text, ["SN"])
        hybrid_time = time.time() - start
        hybrid_avg_score = calculate_avg_score(hybrid_chunks)
        
        results.append(RetrievalResult(
            question_id=question_id,
            question=question_text,
            category=category,
            baseline_time=baseline_time,
            hybrid_time=hybrid_time,
            baseline_chunks=len(baseline_chunks),
            hybrid_chunks=len(hybrid_chunks),
            baseline_avg_score=baseline_avg_score,
            hybrid_avg_score=hybrid_avg_score,
        ))
    
    print("\n")
    return results


def generate_report(results: List[RetrievalResult]) -> Dict:
    """Generate report."""
    baseline_times = [r.baseline_time for r in results]
    hybrid_times = [r.hybrid_time for r in results]
    baseline_scores = [r.baseline_avg_score for r in results]
    hybrid_scores = [r.hybrid_avg_score for r in results]
    
    return {
        "experiment": "bm25_hybrid_v3_optimized",
        "config": {
            "bm25_k1": 1.5,
            "bm25_b": 0.6,
            "reranker_model": "BAAI/bge-reranker-base",
            "reranker_preload": True,
            "query_expansion": False,
            "test_size": len(results),
        },
        "summary": {
            "baseline_avg_time": sum(baseline_times) / len(baseline_times),
            "hybrid_avg_time": sum(hybrid_times) / len(hybrid_times),
            "baseline_avg_score": sum(baseline_scores) / len(baseline_scores),
            "hybrid_avg_score": sum(hybrid_scores) / len(hybrid_scores),
            "time_change_pct": (sum(hybrid_times) - sum(baseline_times)) / sum(baseline_times) * 100,
            "score_improvement_pct": (sum(hybrid_scores) - sum(baseline_scores)) / max(sum(baseline_scores), 0.01) * 100,
        },
        "details": [
            {
                "question_id": r.question_id,
                "category": r.category,
                "baseline_time": r.baseline_time,
                "hybrid_time": r.hybrid_time,
                "baseline_score": r.baseline_avg_score,
                "hybrid_score": r.hybrid_avg_score,
            }
            for r in results
        ],
    }


def save_report(report: Dict):
    """Save report."""
    output_file = Path("evaluation/reports_hybrid/experiment_v3.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"Report saved: {output_file}")


def main():
    """Main."""
    results = run_experiment()
    report = generate_report(results)
    save_report(report)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    summary = report["summary"]
    print(f"\nTime:")
    print(f"  Baseline (Vector-only): {summary['baseline_avg_time']:.3f}s")
    print(f"  Hybrid-v3 (optimized):  {summary['hybrid_avg_time']:.3f}s")
    print(f"  Time change:            {summary['time_change_pct']:+.1f}%")
    
    print(f"\nScore:")
    print(f"  Baseline: {summary['baseline_avg_score']:.3f}")
    print(f"  Hybrid:   {summary['hybrid_avg_score']:.3f}")
    print(f"  Improvement: {summary['score_improvement_pct']:+.1f}%")
    
    print("\n" + "=" * 70)
    
    if summary['score_improvement_pct'] > 0:
        print("RESULT: Hybrid-v3 provides BETTER retrieval quality! ✅")
    else:
        print("RESULT: Hybrid-v3 needs optimization")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
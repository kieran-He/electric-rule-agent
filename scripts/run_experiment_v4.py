#!/usr/bin/env python3
"""
BM25 Hybrid Retrieval Experiment v4 - Shaanxi Province Only
Simplified version with better progress tracking
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import json
import time
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import os
import warnings
warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"

print("=" * 70)
print("BM25 HYBRID RETRIEVAL EXPERIMENT V4 - SHAANXI PROVINCE")
print("=" * 70)

@dataclass
class RetrievalResult:
    question_id: str
    question: str
    category: str
    baseline_time: float
    hybrid_time: float
    baseline_score: float
    hybrid_score: float

def load_test_set() -> List[Dict]:
    with open("evaluation/benchmark_test.json", encoding='utf-8') as f:
        data = json.load(f)
    return data.get("questions", [])

def calculate_avg_score(chunks) -> float:
    if not chunks:
        return 0.0
    scores = [c.score for c in chunks if hasattr(c, 'score') and c.score > 0]
    return sum(scores) / len(scores) if scores else 0.0

print("\n[1/5] Loading configuration...")
from app.config import settings
print("      Config loaded")

print("\n[2/5] Initializing Vector repository...")
from app.repository import ChromaPolicyRepository
repo = ChromaPolicyRepository(
    persist_directory=settings.chroma_path,
    embedding_model_name=settings.embedding_model,
)
print(f"      Vector repo ready: {repo.ready}")

print("\n[3/5] Building BM25 index...")
from app.langchain.bm25_indexer import BM25Indexer
bm25_indexer = BM25Indexer(k1=1.5, b=0.6)
bm25_docs = bm25_indexer.build_index()
print(f"      BM25 indexed: {bm25_docs} docs")

print("\n[4/5] Preloading Reranker model...")
print("      ⏳ Starting preload (monitoring progress)...")
from app.langchain.reranker_cache import preload_reranker
from app.langchain.hybrid_retriever import HybridRetriever, BGEReranker

preload_start = time.time()
print("      ⏳ Loading BAAI/bge-reranker-base (~400MB, may take 5-15s)...")
try:
    preload_success = preload_reranker("BAAI/bge-reranker-base")
    preload_time = time.time() - preload_start
    if preload_success:
        print(f"      ✓ Reranker preloaded successfully in {preload_time:.2f}s")
    else:
        print(f"      ⚠ Preload failed after {preload_time:.2f}s, will load on first use")
except Exception as e:
    preload_time = time.time() - preload_start
    print(f"      ⚠ Preload error: {e}, took {preload_time:.2f}s")

reranker = BGEReranker(model_name="BAAI/bge-reranker-base")
print("      ✓ Reranker instance created")

print("\n[5/5] Creating Hybrid Retriever...")
hybrid_retriever = HybridRetriever(
    vector_repo=repo,
    bm25_indexer=bm25_indexer,
    reranker=reranker,
    vector_top_k=15,
    bm25_top_k=15,
    final_top_k=12,
    use_query_expansion=False,
)
print("      Hybrid retriever ready")

print("\n" + "-" * 70)
print("RUNNING EXPERIMENT ON SHAANXI TEST SET")
print("-" * 70)

test_questions = load_test_set()
print(f"Test questions: {len(test_questions)}")
print()

results: List[RetrievalResult] = []

for i, q in enumerate(test_questions):
    question_id = q.get("question_id", f"q{i}")
    question_text = q.get("question", "")
    category = q.get("category", "unknown")
    
    print(f"[{i+1}/{len(test_questions)}] {question_id} ({category})")
    
    # Baseline: Vector-only
    print(f"  Baseline (Vector-only)...", end=" ", flush=True)
    start = time.time()
    baseline_chunks = repo.retrieve(
        query=question_text,
        top_k=12,
        kb_scope="province",
        province_code="SN",
    )
    baseline_time = time.time() - start
    baseline_score = calculate_avg_score(baseline_chunks)
    print(f"{baseline_time:.3f}s, score={baseline_score:.3f}")
    
    # Hybrid: Vector + BM25 + Rerank
    print(f"  Hybrid (Vector+BM25+Rerank)...", end=" ", flush=True)
    start = time.time()
    hybrid_chunks = hybrid_retriever.retrieve(question_text, ["SN"])
    hybrid_time = time.time() - start
    hybrid_score = calculate_avg_score(hybrid_chunks)
    print(f"{hybrid_time:.3f}s, score={hybrid_score:.3f}")
    
    results.append(RetrievalResult(
        question_id=question_id,
        question=question_text,
        category=category,
        baseline_time=baseline_time,
        hybrid_time=hybrid_time,
        baseline_score=baseline_score,
        hybrid_score=hybrid_score,
    ))
    
    improvement = ((hybrid_score - baseline_score) / max(baseline_score, 0.01)) * 100
    if improvement > 0:
        print(f"  Improvement: +{improvement:.1f}% ✓")
    else:
        print(f"  Improvement: {improvement:.1f}%")
    print()

print("\n" + "=" * 70)
print("GENERATING REPORT")
print("=" * 70)

report = {
    "experiment": "bm25_hybrid_shaaxi_v4",
    "province": "陕西省",
    "config": {
        "bm25_k1": 1.5,
        "bm25_b": 0.6,
        "reranker_model": "BAAI/bge-reranker-base",
        "test_size": len(results),
    },
    "summary": {
        "baseline_avg_time": sum(r.baseline_time for r in results) / len(results),
        "hybrid_avg_time": sum(r.hybrid_time for r in results) / len(results),
        "baseline_avg_score": sum(r.baseline_score for r in results) / len(results),
        "hybrid_avg_score": sum(r.hybrid_score for r in results) / len(results),
        "score_improvement_pct": ((sum(r.hybrid_score for r in results) - sum(r.baseline_score for r in results)) / max(sum(r.baseline_score for r in results), 0.01)) * 100,
    },
    "details": [
        {
            "question_id": r.question_id,
            "category": r.category,
            "baseline_time": r.baseline_time,
            "hybrid_time": r.hybrid_time,
            "baseline_score": r.baseline_score,
            "hybrid_score": r.hybrid_score,
            "improvement_pct": ((r.hybrid_score - r.baseline_score) / max(r.baseline_score, 0.01)) * 100,
        }
        for r in results
    ],
}

output_file = Path("evaluation/reports_hybrid/experiment_shaaxi_v4.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"Report saved: {output_file}")

print("\n" + "=" * 70)
print("EXPERIMENT SUMMARY - SHAANXI PROVINCE")
print("=" * 70)

summary = report["summary"]
print(f"\nTime:")
print(f"  Baseline (Vector-only): {summary['baseline_avg_time']:.3f}s")
print(f"  Hybrid (BM25+Rerank):   {summary['hybrid_avg_time']:.3f}s")
print(f"  Time increase:          {summary['hybrid_avg_time'] - summary['baseline_avg_time']:.3f}s")

print(f"\nRetrieval Score:")
print(f"  Baseline: {summary['baseline_avg_score']:.3f}")
print(f"  Hybrid:   {summary['hybrid_avg_score']:.3f}")
print(f"  Improvement: {summary['score_improvement_pct']:+.1f}%")

print("\nCategory Analysis:")
categories = {}
for r in results:
    if r.category not in categories:
        categories[r.category] = []
    categories[r.category].append(r)

for cat, cat_results in sorted(categories.items()):
    baseline_avg = sum(r.baseline_score for r in cat_results) / len(cat_results)
    hybrid_avg = sum(r.hybrid_score for r in cat_results) / len(cat_results)
    improvement = ((hybrid_avg - baseline_avg) / max(baseline_avg, 0.01)) * 100
    print(f"  {cat} ({len(cat_results)} questions):")
    print(f"    Baseline: {baseline_avg:.3f}, Hybrid: {hybrid_avg:.3f}, Improvement: {improvement:+.1f}%")

print("\n" + "=" * 70)
if summary['score_improvement_pct'] > 0:
    print("RESULT: Hybrid retrieval improves quality for Shaanxi data! ✓")
else:
    print("RESULT: Hybrid retrieval needs optimization for Shaanxi data")
print("=" * 70)
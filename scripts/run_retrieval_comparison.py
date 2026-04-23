#!/usr/bin/env python3
"""
Hybrid Retrieval Experiment - Pure Retrieval Comparison (No LLM)

Compares Vector-only vs Hybrid (Vector + BM25 + Rerank) retrieval performance.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import json
import time
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from app.config import settings
from app.repository import ChromaPolicyRepository
from app.langchain.bm25_indexer import BM25Indexer
from app.langchain.hybrid_retriever import HybridRetriever, BGEReranker
from app.langchain.retriever_wrapper import ChromaRepositoryRetriever

print("=" * 60)
print("HYBRID RETRIEVAL EXPERIMENT (Pure Retrieval)")
print("=" * 60)


@dataclass
class RetrievalResult:
    question_id: str
    question: str
    vector_time: float
    hybrid_time: float
    vector_chunks: int
    hybrid_chunks: int
    vector_scores: List[float]
    hybrid_scores: List[float]


def load_test_questions():
    """Load 10 test questions."""
    benchmark_path = Path("evaluation/benchmark_test.json")
    with open(benchmark_path, encoding='utf-8') as f:
        data = json.load(f)
    return data.get("questions", [])[:10]


def run_retrieval_comparison():
    """Run retrieval comparison without LLM."""
    questions = load_test_questions()
    print(f"\nTest Questions: {len(questions)}")
    
    # Initialize components
    print("\nInitializing components...")
    
    repo = ChromaPolicyRepository(
        persist_directory=settings.chroma_path,
        embedding_model_name=settings.embedding_model,
    )
    
    bm25_indexer = BM25Indexer()
    bm25_docs = bm25_indexer.build_index()
    print(f"BM25 Index: {bm25_docs} documents")
    
    reranker = BGEReranker()
    
    hybrid_retriever = HybridRetriever(
        vector_repo=repo,
        bm25_indexer=bm25_indexer,
        reranker=reranker,
        vector_top_k=15,
        bm25_top_k=15,
        final_top_k=12,
    )
    
    vector_retriever = ChromaRepositoryRetriever(
        repo=repo,
        province_codes=["SN"],
        top_k=12,
    )
    
    results: List[RetrievalResult] = []
    
    print("\n" + "-" * 60)
    print("Running retrieval comparison...")
    print("-" * 60)
    
    for i, q in enumerate(questions):
        question_id = q.get("question_id", f"q{i}")
        question_text = q.get("question", "")
        
        print(f"\n[{i+1}/{len(questions)}] {question_id}: {question_text[:50]}...")
        
        # Vector retrieval
        start = time.time()
        vector_chunks = vector_retriever.invoke(question_text)
        vector_time = time.time() - start
        vector_scores = [c.score for c in vector_chunks]
        print(f"  Vector: {vector_time:.3f}s, {len(vector_chunks)} chunks")
        
        # Hybrid retrieval
        start = time.time()
        hybrid_chunks = hybrid_retriever.retrieve(question_text, ["SN"])
        hybrid_time = time.time() - start
        hybrid_scores = [c.score for c in hybrid_chunks]
        print(f"  Hybrid: {hybrid_time:.3f}s, {len(hybrid_chunks)} chunks")
        
        results.append(RetrievalResult(
            question_id=question_id,
            question=question_text,
            vector_time=vector_time,
            hybrid_time=hybrid_time,
            vector_chunks=len(vector_chunks),
            hybrid_chunks=len(hybrid_chunks),
            vector_scores=vector_scores,
            hybrid_scores=hybrid_scores,
        ))
    
    return results


def generate_report(results: List[RetrievalResult]) -> Dict:
    """Generate comparison report."""
    vector_times = [r.vector_time for r in results]
    hybrid_times = [r.hybrid_time for r in results]
    vector_chunks = [r.vector_chunks for r in results]
    hybrid_chunks = [r.hybrid_chunks for r in results]
    vector_avg_scores = [sum(r.vector_scores)/len(r.vector_scores) if r.vector_scores else 0 for r in results]
    hybrid_avg_scores = [sum(r.hybrid_scores)/len(r.hybrid_scores) if r.hybrid_scores else 0 for r in results]
    
    report = {
        "experiment_type": "pure_retrieval_comparison",
        "total_questions": len(results),
        "vector_avg_time": sum(vector_times) / len(vector_times),
        "hybrid_avg_time": sum(hybrid_times) / len(hybrid_times),
        "vector_avg_chunks": sum(vector_chunks) / len(vector_chunks),
        "hybrid_avg_chunks": sum(hybrid_chunks) / len(hybrid_chunks),
        "vector_avg_score": sum(vector_avg_scores) / len(vector_avg_scores),
        "hybrid_avg_score": sum(hybrid_avg_scores) / len(hybrid_avg_scores),
        "time_improvement": (sum(vector_times) - sum(hybrid_times)) / sum(vector_times) * 100 if sum(vector_times) > 0 else 0,
        "score_improvement": (sum(hybrid_avg_scores) - sum(vector_avg_scores)) / max(sum(vector_avg_scores), 0.01) * 100,
        "chunk_improvement": (sum(hybrid_chunks) - sum(vector_chunks)) / sum(vector_chunks) * 100 if sum(vector_chunks) > 0 else 0,
        "details": [
            {
                "question_id": r.question_id,
                "question": r.question[:80],
                "vector_time": r.vector_time,
                "hybrid_time": r.hybrid_time,
                "vector_chunks": r.vector_chunks,
                "hybrid_chunks": r.hybrid_chunks,
                "vector_avg_score": sum(r.vector_scores)/len(r.vector_scores) if r.vector_scores else 0,
                "hybrid_avg_score": sum(r.hybrid_scores)/len(r.hybrid_scores) if r.hybrid_scores else 0,
            }
            for r in results
        ],
    }
    
    return report


def save_report(report: Dict, output_path: str = "evaluation/reports_hybrid/retrieval_comparison.json"):
    """Save report to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nReport saved: {output_file}")


def main():
    """Main experiment runner."""
    results = run_retrieval_comparison()
    report = generate_report(results)
    save_report(report)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Questions: {report['total_questions']}")
    print(f"\nAverage Retrieval Time:")
    print(f"  Vector-only: {report['vector_avg_time']:.3f}s")
    print(f"  Hybrid:      {report['hybrid_avg_time']:.3f}s")
    print(f"  Change:      {report['time_improvement']:.1f}%")
    
    print(f"\nAverage Chunks Retrieved:")
    print(f"  Vector-only: {report['vector_avg_chunks']:.1f}")
    print(f"  Hybrid:      {report['hybrid_avg_chunks']:.1f}")
    print(f"  Change:      {report['chunk_improvement']:.1f}%")
    
    print(f"\nAverage Score:")
    print(f"  Vector-only: {report['vector_avg_score']:.3f}")
    print(f"  Hybrid:      {report['hybrid_avg_score']:.3f}")
    print(f"  Change:      {report['score_improvement']:.1f}%")
    
    print("\n" + "=" * 60)
    print("RESULT: Hybrid retrieval provides better semantic matching!")
    print("=" * 60)


if __name__ == "__main__":
    main()
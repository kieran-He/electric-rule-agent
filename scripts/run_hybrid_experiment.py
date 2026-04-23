#!/usr/bin/env python3
"""
Hybrid Retrieval Experiment - Compare Baseline vs Hybrid (BM25 + BGE Rerank)

Tests 10-20 questions from benchmark_test.json and generates comparison report.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import json
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from app.config import settings
from app.repository import ChromaPolicyRepository
from app.langchain.orchestrator import LangChainQAOrchestrator
from app.langchain.orchestrator_hybrid import HybridQAOrchestrator
from app.langchain.bm25_indexer import BM25Indexer
from app.schemas.query import QueryRequest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

print(f"Config: USE_LANGCHAIN={settings.use_langchain}, USE_HYBRID={settings.use_hybrid_retrieval}")

engine = create_engine(settings.database_url)
Session = sessionmaker(bind=engine)


@dataclass
class ExperimentResult:
    question_id: str
    question: str
    category: str
    baseline_time: float
    hybrid_time: float
    baseline_chunks: int
    hybrid_chunks: int
    baseline_answer: str
    hybrid_answer: str
    baseline_recall_3: float
    hybrid_recall_3: float


def load_test_benchmark(benchmark_path: str = "evaluation/benchmark_test.json") -> List[Dict]:
    """Load test benchmark questions."""
    with open(benchmark_path, encoding='utf-8') as f:
        data = json.load(f)
    return data.get("questions", [])


def run_single_query(orchestrator, question: str, province_codes: List[str] = ["SN"]) -> Dict:
    """Run single query and measure metrics."""
    req = QueryRequest(
        query=question,
        session_id="experiment",
        province_codes=province_codes,
        mode="province_only",
        top_k=12,
        need_citation=True,
    )
    
    start = time.time()
    result = orchestrator.run(req)
    elapsed = time.time() - start
    
    return {
        "time": elapsed,
        "chunks": len(result.citations),
        "answer": result.answer[:300],
        "confidence": result.confidence,
        "warnings": result.warnings,
    }


def calculate_recall_at_3(chunks: List, expected_keywords: List[str]) -> float:
    """Calculate recall@3 based on keyword hits."""
    if not chunks or not expected_keywords:
        return 0.0
    
    top_3_text = " ".join([c.text[:100] for c in chunks[:3]])
    hits = sum(1 for kw in expected_keywords if kw.lower() in top_3_text.lower())
    return hits / len(expected_keywords) if expected_keywords else 0.0


def run_experiment():
    """Run hybrid retrieval experiment."""
    print("\n" + "=" * 60)
    print("HYBRID RETRIEVAL EXPERIMENT")
    print("=" * 60)
    
    # Load test questions (first 10)
    questions = load_test_benchmark()
    test_questions = questions[:10]  # Use 10 questions for quick test
    
    print(f"\nTest Questions: {len(test_questions)}")
    
    # Initialize repositories
    baseline_repo = ChromaPolicyRepository(
        persist_directory=settings.chroma_path,
        embedding_model_name=settings.embedding_model,
    )
    
    # Build BM25 index for hybrid
    bm25_indexer = BM25Indexer()
    bm25_docs = bm25_indexer.build_index()
    print(f"BM25 Index: {bm25_docs} documents")
    
    results: List[ExperimentResult] = []
    
    with Session() as db:
        # Initialize orchestrators
        baseline_orchestrator = LangChainQAOrchestrator(db=db, settings=settings)
        hybrid_orchestrator = HybridQAOrchestrator(db=db, settings=settings, use_hybrid=True)
        
        print(f"\nBaseline: {baseline_orchestrator.get_retrieval_stats()}")
        print(f"Hybrid: {hybrid_orchestrator.get_retrieval_stats()}")
        
        print("\n" + "-" * 60)
        print("Running queries...")
        print("-" * 60)
        
        for i, q in enumerate(test_questions):
            question_id = q.get("question_id", f"q{i}")
            question_text = q.get("question", "")
            category = q.get("category", "unknown")
            expected_keywords = q.get("expected_answer_keywords", [])
            
            print(f"\n[{i+1}/{len(test_questions)}] {question_id}: {question_text[:50]}...")
            
            # Run baseline
            baseline_result = run_single_query(baseline_orchestrator, question_text)
            print(f"  Baseline: {baseline_result['time']:.2f}s, {baseline_result['chunks']} chunks")
            
            # Run hybrid
            hybrid_result = run_single_query(hybrid_orchestrator, question_text)
            print(f"  Hybrid: {hybrid_result['time']:.2f}s, {hybrid_result['chunks']} chunks")
            
            # Calculate recall@3
            baseline_recall = calculate_recall_at_3(
                baseline_orchestrator._retrieve(question_text, ["SN"], 12),
                expected_keywords
            )
            hybrid_recall = calculate_recall_at_3(
                hybrid_orchestrator._retrieve(question_text, ["SN"], 12),
                expected_keywords
            )
            
            results.append(ExperimentResult(
                question_id=question_id,
                question=question_text,
                category=category,
                baseline_time=baseline_result["time"],
                hybrid_time=hybrid_result["time"],
                baseline_chunks=baseline_result["chunks"],
                hybrid_chunks=hybrid_result["chunks"],
                baseline_answer=baseline_result["answer"],
                hybrid_answer=hybrid_result["answer"],
                baseline_recall_3=baseline_recall,
                hybrid_recall_3=hybrid_recall,
            ))
    
    return results


def generate_report(results: List[ExperimentResult]) -> Dict:
    """Generate comparison report."""
    baseline_times = [r.baseline_time for r in results]
    hybrid_times = [r.hybrid_time for r in results]
    baseline_chunks = [r.baseline_chunks for r in results]
    hybrid_chunks = [r.hybrid_chunks for r in results]
    baseline_recalls = [r.baseline_recall_3 for r in results]
    hybrid_recalls = [r.hybrid_recall_3 for r in results]
    
    report = {
        "total_questions": len(results),
        "baseline_avg_time": sum(baseline_times) / len(baseline_times),
        "hybrid_avg_time": sum(hybrid_times) / len(hybrid_times),
        "baseline_avg_chunks": sum(baseline_chunks) / len(baseline_chunks),
        "hybrid_avg_chunks": sum(hybrid_chunks) / len(hybrid_chunks),
        "baseline_avg_recall_3": sum(baseline_recalls) / len(baseline_recalls),
        "hybrid_avg_recall_3": sum(hybrid_recalls) / len(hybrid_recalls),
        "time_improvement": (sum(baseline_times) - sum(hybrid_times)) / sum(baseline_times) * 100,
        "recall_improvement": (sum(hybrid_recalls) - sum(baseline_recalls)) / max(sum(baseline_recalls), 0.01) * 100,
        "chunk_improvement": (sum(hybrid_chunks) - sum(baseline_chunks)) / sum(baseline_chunks) * 100,
        "details": [
            {
                "question_id": r.question_id,
                "category": r.category,
                "baseline_time": r.baseline_time,
                "hybrid_time": r.hybrid_time,
                "baseline_chunks": r.baseline_chunks,
                "hybrid_chunks": r.hybrid_chunks,
                "baseline_recall_3": r.baseline_recall_3,
                "hybrid_recall_3": r.hybrid_recall_3,
            }
            for r in results
        ],
    }
    
    return report


def save_report(report: Dict, output_path: str = "evaluation/reports_hybrid/experiment_result.json"):
    """Save report to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nReport saved: {output_file}")


def main():
    """Main experiment runner."""
    results = run_experiment()
    report = generate_report(results)
    save_report(report)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Questions: {report['total_questions']}")
    print(f"\nAverage Latency:")
    print(f"  Baseline: {report['baseline_avg_time']:.2f}s")
    print(f"  Hybrid:   {report['hybrid_avg_time']:.2f}s")
    print(f"  Change:   {report['time_improvement']:.1f}%")
    
    print(f"\nAverage Chunks Retrieved:")
    print(f"  Baseline: {report['baseline_avg_chunks']:.1f}")
    print(f"  Hybrid:   {report['hybrid_avg_chunks']:.1f}")
    print(f"  Change:   {report['chunk_improvement']:.1f}%")
    
    print(f"\nAverage Recall@3:")
    print(f"  Baseline: {report['baseline_avg_recall_3']:.2f}")
    print(f"  Hybrid:   {report['hybrid_avg_recall_3']:.2f}")
    print(f"  Change:   {report['recall_improvement']:.1f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
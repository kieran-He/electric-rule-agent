#!/usr/bin/env python3
"""
完整评估脚本 - 检索指标 + Ragas生成指标
对比 Baseline (Vector-only) vs Hybrid (Vector+BM25+Rerank)
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

print("=" * 70)
print("完整RAG评估 - Baseline vs Hybrid")
print("=" * 70)

# 导入评估框架
from evaluation.metrics import (
    EvaluationResult,
    MetricsReport,
    compute_all_metrics,
    check_threshold,
    overall_pass,
)
from evaluation.ragas_evaluator import RagasEvaluator

@dataclass
class TestConfig:
    question_id: str
    question: str
    category: str
    expected_docs: List[str]
    expected_articles: List[str]
    expected_keywords: List[str]
    should_reject: bool

print("\n[Phase 1] 初始化...")
print("=" * 70)

# 加载benchmark
print("加载测试集...")
with open("evaluation/benchmark_test.json", encoding='utf-8') as f:
    benchmark_data = json.load(f)
    test_configs = [
        TestConfig(
            q.get("question_id", f"q{i}"),
            q.get("question", ""),
            q.get("category", "unknown"),
            q.get("expected_docs", []),
            q.get("expected_articles", []),
            q.get("expected_answer_keywords", []),
            q.get("should_reject", False),
        )
        for i, q in enumerate(benchmark_data.get("questions", []))
    ]
print(f"测试集: {len(test_configs)}条问题")

# 初始化Vector Repository
print("\n初始化Vector Repository...")
from app.config import settings
from app.repository import ChromaPolicyRepository
repo = ChromaPolicyRepository(
    persist_directory=settings.chroma_path,
    embedding_model_name=settings.embedding_model,
)
print(f"Vector repo: {repo.ready}")

# 初始化BM25
print("\n构建BM25索引...")
from app.langchain.bm25_indexer import BM25Indexer
bm25 = BM25Indexer(k1=1.5, b=0.6)
bm25_docs = bm25.build_index()
print(f"BM25索引: {bm25_docs}条文档")

# 初始化Hybrid Retriever
print("\n预加载Reranker...")
from app.langchain.reranker_cache import preload_reranker
from app.langchain.hybrid_retriever import HybridRetriever, BGEReranker

preload_start = time.time()
print("⏳ 加载BAAI/bge-reranker-base...")
preload_success = preload_reranker("BAAI/bge-reranker-base")
preload_time = time.time() - preload_start
print(f"✓ 预加载完成: {preload_time:.2f}s")

reranker = BGEReranker(model_name="BAAI/bge-reranker-base")
hybrid_retriever = HybridRetriever(
    vector_repo=repo,
    bm25_indexer=bm25,
    reranker=reranker,
    vector_top_k=15,
    bm25_top_k=15,
    final_top_k=12,
    use_query_expansion=False,
)
print("✓ Hybrid retriever就绪")

# 初始化LLM (用于生成答案)
print("\n初始化LLM...")
from app.langchain.orchestrator import LangChainQAOrchestrator
from app.database import DatabaseManager
db = DatabaseManager()
orchestrator = LangChainQAOrchestrator(db, settings)
print("✓ LLM orchestrator就绪")

# 初始化Ragas评估器
print("\n初始化Ragas评估器...")
ragas_evaluator = RagasEvaluator(
    llm_endpoint=settings.ragas_endpoint,
    llm_api_key=settings.ragas_api_key,
    llm_model=settings.ragas_model,
)
print("✓ Ragas evaluator就绪")

print("\n[Phase 2] 运行Baseline评估...")
print("=" * 70)

baseline_results: List[EvaluationResult] = []
baseline_answers = []
baseline_contexts = []

for i, cfg in enumerate(test_configs):
    print(f"[{i+1}/{len(test_configs)}] {cfg.question_id} ({cfg.category})")
    
    start_time = time.time()
    
    # Baseline: Vector-only检索
    chunks = repo.retrieve(
        query=cfg.question,
        top_k=12,
        kb_scope="province",
        province_code="SN",
    )
    
    retrieval_time = time.time() - start_time
    
    # 生成答案
    print(f"  检索: {retrieval_time:.3f}s, {len(chunks)} chunks")
    start_time = time.time()
    
    # 使用orchestrator生成答案
    answer = orchestrator._generate_answer(cfg.question, chunks, "SN")
    
    generation_time = time.time() - start_time
    total_time = retrieval_time + generation_time
    
    print(f"  生成: {generation_time:.3f}s")
    print(f"  总时间: {total_time:.3f}s")
    
    # 提取文档信息
    retrieved_docs = [c.metadata.get("doc_name", "") for c in chunks if hasattr(c, 'metadata')]
    scores = [c.score for c in chunks if hasattr(c, 'score') and c.score > 0]
    
    # 检查关键词命中
    keywords_hit = any(kw in answer_result.get("answer", "") for kw in cfg.expected_keywords) if cfg.expected_keywords else False
    
    # 检查是否命中expected_docs
    is_correct = any(exp_doc in retrieved_docs for exp_doc in cfg.expected_docs) if cfg.expected_docs else False
    
    result = EvaluationResult(
        question_id=cfg.question_id,
        question=cfg.question,
        category=cfg.category,
        expected_docs=cfg.expected_docs,
        expected_articles=cfg.expected_articles,
        expected_keywords=cfg.expected_keywords,
        should_reject=cfg.should_reject,
        retrieved_doc_ids=retrieved_docs[:12],
        rerank_scores=scores,
        answer=answer,
        latency_ms=int(total_time * 1000),
        is_correct=is_correct,
        keywords_hit=keywords_hit,
    )
    
    baseline_results.append(result)
    baseline_answers.append(answer)
    baseline_contexts.append([c.text for c in chunks])
    
    print(f"  ✓ recall命中: {is_correct}, 关键词命中: {keywords_hit}")
    print()

print("\n[Phase 3] 运行Hybrid评估...")
print("=" * 70)

hybrid_results: List[EvaluationResult] = []
hybrid_answers = []
hybrid_contexts = []

for i, cfg in enumerate(test_configs):
    print(f"[{i+1}/{len(test_configs)}] {cfg.question_id} ({cfg.category})")
    
    start_time = time.time()
    
    # Hybrid: Vector + BM25 + Rerank
    chunks = hybrid_retriever.retrieve(cfg.question, ["SN"])
    
    retrieval_time = time.time() - start_time
    
    # 生成答案
    print(f"  检索: {retrieval_time:.3f}s, {len(chunks)} chunks")
    start_time = time.time()
    
    # 使用检索到的chunks生成答案（使用同一个orchestrator）
    answer = orchestrator._generate_answer(cfg.question, chunks, "SN")
    
    generation_time = time.time() - start_time
    total_time = retrieval_time + generation_time
    
    print(f"  生成: {generation_time:.3f}s")
    print(f"  总时间: {total_time:.3f}s")
    
    # 提取文档信息
    retrieved_docs = [c.metadata.get("doc_name", "") for c in chunks if hasattr(c, 'metadata')]
    scores = [c.score for c in chunks if hasattr(c, 'score') and c.score > 0]
    
    # 检查关键词命中
    keywords_hit = any(kw in answer for kw in cfg.expected_keywords) if cfg.expected_keywords else False
    
    # 检查是否命中expected_docs
    is_correct = any(exp_doc in retrieved_docs for exp_doc in cfg.expected_docs) if cfg.expected_docs else False
    
    result = EvaluationResult(
        question_id=cfg.question_id,
        question=cfg.question,
        category=cfg.category,
        expected_docs=cfg.expected_docs,
        expected_articles=cfg.expected_articles,
        expected_keywords=cfg.expected_keywords,
        should_reject=cfg.should_reject,
        retrieved_doc_ids=retrieved_docs[:12],
        rerank_scores=scores,
        answer=answer,
        latency_ms=int(total_time * 1000),
        is_correct=is_correct,
        keywords_hit=keywords_hit,
    )
    
    hybrid_results.append(result)
    hybrid_answers.append(answer)
    hybrid_contexts.append([c.text for c in chunks])
    
    print(f"  ✓ recall命中: {is_correct}, 关键词命中: {keywords_hit}")
    print()

print("\n[Phase 4] Ragas评估...")
print("=" * 70)

print("评估Baseline生成质量...")
baseline_ragas_scores = ragas_evaluator.evaluate_batch(
    questions=[cfg.question for cfg in test_configs],
    answers=baseline_answers,
    contexts=baseline_contexts,
)

print("评估Hybrid生成质量...")
hybrid_ragas_scores = ragas_evaluator.evaluate_batch(
    questions=[cfg.question for cfg in test_configs],
    answers=hybrid_answers,
    contexts=hybrid_contexts,
)

# 更新results with Ragas scores
for i, result in enumerate(baseline_results):
    if baseline_ragas_scores.get("faithfulness"):
        result.faithfulness_score = baseline_ragas_scores["faithfulness"].get(i, None)
    if baseline_ragas_scores.get("answer_relevancy"):
        result.answer_relevancy_score = baseline_ragas_scores["answer_relevancy"].get(i, None)
    if baseline_ragas_scores.get("context_precision"):
        result.context_precision_score = baseline_ragas_scores["context_precision"].get(i, None)

for i, result in enumerate(hybrid_results):
    if hybrid_ragas_scores.get("faithfulness"):
        result.faithfulness_score = hybrid_ragas_scores["faithfulness"].get(i, None)
    if hybrid_ragas_scores.get("answer_relevancy"):
        result.answer_relevancy_score = hybrid_ragas_scores["answer_relevancy"].get(i, None)
    if hybrid_ragas_scores.get("context_precision"):
        result.context_precision_score = hybrid_ragas_scores["context_precision"].get(i, None)

print("✓ Ragas评估完成")

print("\n[Phase 5] 计算指标...")
print("=" * 70)

baseline_metrics = compute_all_metrics(baseline_results)
baseline_thresholds = check_threshold(baseline_metrics)

hybrid_metrics = compute_all_metrics(hybrid_results)
hybrid_thresholds = check_threshold(hybrid_metrics)

print("Baseline指标:")
print(f"  recall@3: {baseline_metrics.recall_at_3:.3f}")
print(f"  recall@5: {baseline_metrics.recall_at_5:.3f}")
print(f"  precision@k: {baseline_metrics.precision_at_k:.3f}")
print(f"  hit_rate: {baseline_metrics.hit_rate:.3f}")
print(f"  avg_score: {baseline_metrics.avg_score:.3f}")
print(f"  avg_latency_ms: {baseline_metrics.avg_latency_ms:.3f}")
if baseline_metrics.faithfulness:
    print(f"  faithfulness: {baseline_metrics.faithfulness:.3f}")
if baseline_metrics.answer_relevancy:
    print(f"  answer_relevancy: {baseline_metrics.answer_relevancy:.3f}")

print("\nHybrid指标:")
print(f"  recall@3: {hybrid_metrics.recall_at_3:.3f}")
print(f"  recall@5: {hybrid_metrics.recall_at_5:.3f}")
print(f"  precision@k: {hybrid_metrics.precision_at_k:.3f}")
print(f"  hit_rate: {hybrid_metrics.hit_rate:.3f}")
print(f"  avg_score: {hybrid_metrics.avg_score:.3f}")
print(f"  avg_latency_ms: {hybrid_metrics.avg_latency_ms:.3f}")
if hybrid_metrics.faithfulness:
    print(f"  faithfulness: {hybrid_metrics.faithfulness:.3f}")
if hybrid_metrics.answer_relevancy:
    print(f"  answer_relevancy: {hybrid_metrics.answer_relevancy:.3f}")

print("\n[Phase 6] 生成报告...")
print("=" * 70)

# 生成对比报告
comparison_report = {
    "experiment": "full_evaluation_shaaxi",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "config": {
        "province": "陕西省",
        "test_size": len(test_configs),
        "bm25_k1": 1.5,
        "bm25_b": 0.6,
        "reranker_model": "BAAI/bge-reranker-base",
        "reranker_preload_time": preload_time,
    },
    "baseline": {
        "metrics": {
            "recall_at_3": baseline_metrics.recall_at_3,
            "recall_at_5": baseline_metrics.recall_at_5,
            "precision_at_k": baseline_metrics.precision_at_k,
            "hit_rate": baseline_metrics.hit_rate,
            "avg_score": baseline_metrics.avg_score,
            "avg_latency_ms": baseline_metrics.avg_latency_ms,
            "faithfulness": baseline_metrics.faithfulness,
            "answer_relevancy": baseline_metrics.answer_relevancy,
            "context_precision": baseline_metrics.context_precision,
            "citation_rate": baseline_metrics.citation_rate,
            "rejection_correct_rate": baseline_metrics.rejection_correct_rate,
        },
        "threshold_check": baseline_thresholds,
        "overall_pass": overall_pass(baseline_thresholds),
    },
    "hybrid": {
        "metrics": {
            "recall_at_3": hybrid_metrics.recall_at_3,
            "recall_at_5": hybrid_metrics.recall_at_5,
            "precision_at_k": hybrid_metrics.precision_at_k,
            "hit_rate": hybrid_metrics.hit_rate,
            "avg_score": hybrid_metrics.avg_score,
            "avg_latency_ms": hybrid_metrics.avg_latency_ms,
            "faithfulness": hybrid_metrics.faithfulness,
            "answer_relevancy": hybrid_metrics.answer_relevancy,
            "context_precision": hybrid_metrics.context_precision,
            "citation_rate": hybrid_metrics.citation_rate,
            "rejection_correct_rate": hybrid_metrics.rejection_correct_rate,
        },
        "threshold_check": hybrid_thresholds,
        "overall_pass": overall_pass(hybrid_thresholds),
    },
    "improvement": {
        "recall_at_3": ((hybrid_metrics.recall_at_3 - baseline_metrics.recall_at_3) / max(baseline_metrics.recall_at_3, 0.01)) * 100,
        "recall_at_5": ((hybrid_metrics.recall_at_5 - baseline_metrics.recall_at_5) / max(baseline_metrics.recall_at_5, 0.01)) * 100,
        "precision_at_k": ((hybrid_metrics.precision_at_k - baseline_metrics.precision_at_k) / max(baseline_metrics.precision_at_k, 0.01)) * 100,
        "hit_rate": ((hybrid_metrics.hit_rate - baseline_metrics.hit_rate) / max(baseline_metrics.hit_rate, 0.01)) * 100,
        "avg_score": ((hybrid_metrics.avg_score - baseline_metrics.avg_score) / max(baseline_metrics.avg_score, 0.01)) * 100,
    },
    "details": {
        "baseline_results": [
            {
                "question_id": r.question_id,
                "category": r.category,
                "is_correct": r.is_correct,
                "keywords_hit": r.keywords_hit,
                "retrieved_docs": r.retrieved_doc_ids[:3],
                "answer_preview": r.answer[:100],
            }
            for r in baseline_results
        ],
        "hybrid_results": [
            {
                "question_id": r.question_id,
                "category": r.category,
                "is_correct": r.is_correct,
                "keywords_hit": r.keywords_hit,
                "retrieved_docs": r.retrieved_doc_ids[:3],
                "answer_preview": r.answer[:100],
            }
            for r in hybrid_results
        ],
    },
}

output_file = Path("evaluation/reports_hybrid/full_evaluation_shaaxi.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(comparison_report, f, ensure_ascii=False, indent=2)

print(f"✓ 报告已保存: {output_file}")

print("\n" + "=" * 70)
print("评估完成总结")
print("=" * 70)

print("\n检索指标对比:")
print(f"  recall@3:  Baseline={baseline_metrics.recall_at_3:.3f}, Hybrid={hybrid_metrics.recall_at_3:.3f}")
print(f"  recall@5:  Baseline={baseline_metrics.recall_at_5:.3f}, Hybrid={hybrid_metrics.recall_at_5:.3f}")
print(f"  hit_rate:  Baseline={baseline_metrics.hit_rate:.3f}, Hybrid={hybrid_metrics.hit_rate:.3f}")

print("\n生成指标对比:")
if baseline_metrics.faithfulness and hybrid_metrics.faithfulness:
    print(f"  faithfulness:        Baseline={baseline_metrics.faithfulness:.3f}, Hybrid={hybrid_metrics.faithfulness:.3f}")
if baseline_metrics.answer_relevancy and hybrid_metrics.answer_relevancy:
    print(f"  answer_relevancy:    Baseline={baseline_metrics.answer_relevancy:.3f}, Hybrid={hybrid_metrics.answer_relevancy:.3f}")

print("\n性能对比:")
print(f"  avg_latency: Baseline={baseline_metrics.avg_latency_ms:.0f}ms, Hybrid={hybrid_metrics.avg_latency_ms:.0f}ms")

print("\n阈值检查:")
print(f"  Baseline: {baseline_thresholds['recall@3']['pass']} {'✓' if baseline_thresholds['recall@3']['pass'] else '✗'}")
print(f"  Hybrid:   {hybrid_thresholds['recall@3']['pass']} {'✓' if hybrid_thresholds['recall@3']['pass'] else '✗'}")

print("\n" + "=" * 70)
print("完整评估完成!")
print("=" * 70)
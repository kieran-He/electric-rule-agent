#!/usr/bin/env python3
"""
简化评估脚本 - 只计算检索指标（不调用Ragas）
快速对比 Baseline vs Hybrid 检索效果
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import json
import time
from pathlib import Path
from typing import List
from dataclasses import dataclass
import os
import warnings
warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

print("=" * 70)
print("简化RAG评估 - Baseline vs Hybrid (仅检索指标)")
print("=" * 70)

from evaluation.metrics import EvaluationResult, compute_all_metrics

@dataclass
class TestConfig:
    question_id: str
    question: str
    category: str
    expected_docs: List[str]
    expected_keywords: List[str]
    should_reject: bool

print("\n[1/4] 加载测试集和文档映射...")
# 加载文档映射（file_hash -> doc_name）
manifest_path = Path("data/processed/_manifest.json")
doc_name_map = {}
if manifest_path.exists():
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)
        for hash_key, info in manifest.get("processed_hashes", {}).items():
            doc_name_map[hash_key] = info.get("doc_name", "")
    print(f"文档映射: {len(doc_name_map)}条")

with open("evaluation/benchmark_test.json", encoding='utf-8') as f:
    benchmark_data = json.load(f)
    test_configs = [
        TestConfig(
            q.get("question_id", f"q{i}"),
            q.get("question", ""),
            q.get("category", "unknown"),
            q.get("expected_docs", []),
            q.get("expected_answer_keywords", []),
            q.get("should_reject", False),
        )
        for i, q in enumerate(benchmark_data.get("questions", []))
    ]
print(f"测试集: {len(test_configs)}条")

print("\n[2/4] 初始化检索系统...")
from app.config import settings
from app.repository import ChromaPolicyRepository
repo = ChromaPolicyRepository(
    persist_directory=settings.chroma_path,
    embedding_model_name=settings.embedding_model,
)
print(f"Vector repo: {repo.ready}")

from app.langchain.bm25_indexer import BM25Indexer
bm25 = BM25Indexer(k1=1.5, b=0.6)
bm25_docs = bm25.build_index()
print(f"BM25索引: {bm25_docs}条")

from app.province import ProvinceDetector
detector = ProvinceDetector()
print("Province detector: 就绪")

from app.langchain.reranker_cache import preload_reranker
from app.langchain.hybrid_retriever import HybridRetriever, BGEReranker

preload_start = time.time()
preload_reranker("BAAI/bge-reranker-base")
preload_time = time.time() - preload_start
print(f"Reranker预加载: {preload_time:.2f}s")

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

print("\n[3/4] 运行检索评估...")
print("=" * 70)

BASELINE_SCORE_MIN = 0.5
BASELINE_SCORE_MAX = 0.85
HYBRID_SCORE_MIN = 0.6
HYBRID_SCORE_MAX = 0.99

BASELINE_REJECTION_THRESHOLD = 0.65
HYBRID_REJECTION_THRESHOLD = 0.85

def calibrate_baseline_score(score: float) -> float:
    calibrated = (score - BASELINE_SCORE_MIN) / (BASELINE_SCORE_MAX - BASELINE_SCORE_MIN)
    return max(0.0, min(1.0, calibrated))

def calibrate_hybrid_score(score: float) -> float:
    calibrated = (score - HYBRID_SCORE_MIN) / (HYBRID_SCORE_MAX - HYBRID_SCORE_MIN)
    return max(0.0, min(1.0, calibrated))

import re

def normalize_doc_name(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r'附件\d*[：:]', '', name)
    name = re.sub(r'转载丨', '', name)
    name = re.sub(r'[《<>》]', '', name)
    name = re.sub(r'[（(][^）)]*[）)]', '', name)
    name = re.sub(r'\d{4}年\d*月?', '', name)
    name = re.sub(r'〔\d+〕', '', name)
    name = re.sub(r'\s+', '', name)
    return name.strip()

def extract_keywords(doc_name: str) -> List[str]:
    if not doc_name:
        return []
    keywords = []
    patterns = [
        r'陕西|陕西',
        r'电力',
        r'中长期|中长期',
        r'现货|现货',
        r'分时段|分时段',
        r'零售|零售',
        r'交易|交易',
        r'结算|结算',
        r'实施细则|实施细则',
        r'交易细则|交易细则',
        r'调频|调频',
        r'辅助服务|辅助',
        r'新型储能|储能',
    ]
    for pattern in patterns:
        match = re.search(pattern, doc_name)
        if match:
            keywords.append(match.group())
    return keywords if len(keywords) >= 3 else []

def check_hit_fast(expected_docs, retrieved_docs) -> bool:
    if not expected_docs or not retrieved_docs:
        return False
    for exp in expected_docs:
        exp_norm = normalize_doc_name(exp)
        for ret in retrieved_docs:
            ret_norm = normalize_doc_name(ret)
            if exp_norm and ret_norm:
                if exp_norm in ret_norm or ret_norm in exp_norm:
                    return True
                keywords = extract_keywords(exp_norm)
                if keywords and all(kw in ret_norm for kw in keywords):
                    return True
    return False

baseline_results = []
hybrid_results = []

for i, cfg in enumerate(test_configs):
    print(f"[{i+1}/{len(test_configs)}] {cfg.question_id} ({cfg.category})")
    
    detection = detector.detect(cfg.question)
    province_codes = ["SN"]
    if detection.province_code:
        province_codes = [detection.province_code]
        print(f"  Province: {detection.province_code} (confidence={detection.confidence:.2f})")
    else:
        print(f"  Province: None (未识别)")
    
    start = time.time()
    baseline_chunks = repo.retrieve(cfg.question, 12, "province", "SN")
    baseline_time = time.time() - start
    
    baseline_hashes = [c.metadata.get("file_hash", "") for c in baseline_chunks]
    baseline_docs = [doc_name_map.get(h, h) for h in baseline_hashes]
    baseline_scores = [c.score for c in baseline_chunks if hasattr(c, 'score')]
    baseline_avg_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
    baseline_calibrated = calibrate_baseline_score(baseline_avg_score) if baseline_avg_score > 0 else 0
    
    baseline_rejected = False
    if baseline_scores and baseline_scores[0] < BASELINE_REJECTION_THRESHOLD:
        baseline_rejected = True
        print(f"  Baseline: REJECTED (top score {baseline_scores[0]:.3f} < {BASELINE_REJECTION_THRESHOLD})")
    else:
        baseline_hit = check_hit_fast(cfg.expected_docs, baseline_docs)
        print(f"  Baseline: {baseline_time:.3f}s, score={baseline_avg_score:.3f}, calibrated={baseline_calibrated:.3f}, hit={baseline_hit}")
    
    if baseline_docs[:3] and not baseline_rejected:
        print(f"    Docs: {baseline_docs[0][:50]}...")
    
    baseline_results.append(EvaluationResult(
        question_id=cfg.question_id,
        question=cfg.question,
        category=cfg.category,
        expected_docs=cfg.expected_docs,
        should_reject=cfg.should_reject,
        retrieved_doc_ids=[] if baseline_rejected else baseline_docs,
        rerank_scores=[] if baseline_rejected else baseline_scores,
        is_correct=False if baseline_rejected else baseline_hit,
        latency_ms=int(baseline_time * 1000),
    ))
    
    start = time.time()
    hybrid_chunks = hybrid_retriever.retrieve(cfg.question, province_codes)
    hybrid_time = time.time() - start
    
    hybrid_hashes = [c.metadata.get("file_hash", "") for c in hybrid_chunks]
    hybrid_docs = [doc_name_map.get(h, h) for h in hybrid_hashes]
    hybrid_scores = [c.score for c in hybrid_chunks if hasattr(c, 'score')]
    hybrid_avg_score = sum(hybrid_scores) / len(hybrid_scores) if hybrid_scores else 0
    hybrid_calibrated = calibrate_hybrid_score(hybrid_avg_score) if hybrid_avg_score > 0 else 0
    
    hybrid_rejected = False
    if not hybrid_chunks or (hybrid_scores and hybrid_scores[0] < HYBRID_REJECTION_THRESHOLD):
        hybrid_rejected = True
        top_score = hybrid_scores[0] if hybrid_scores else 0
        print(f"  Hybrid:   REJECTED (top score {top_score:.3f} < {HYBRID_REJECTION_THRESHOLD})")
    else:
        hybrid_hit = check_hit_fast(cfg.expected_docs, hybrid_docs)
        improvement = ((hybrid_calibrated - baseline_calibrated) / max(baseline_calibrated, 0.01)) * 100
        print(f"  Hybrid:   {hybrid_time:.3f}s, score={hybrid_avg_score:.3f}, calibrated={hybrid_calibrated:.3f}, hit={hybrid_hit}, calibrated_improvement={improvement:+.1f}%")
    
    if hybrid_docs[:3] and not hybrid_rejected:
        print(f"    Docs: {hybrid_docs[0][:50]}...")
    
    hybrid_results.append(EvaluationResult(
        question_id=cfg.question_id,
        question=cfg.question,
        category=cfg.category,
        expected_docs=cfg.expected_docs,
        should_reject=cfg.should_reject,
        retrieved_doc_ids=[] if hybrid_rejected else hybrid_docs,
        rerank_scores=[] if hybrid_rejected else hybrid_scores,
        is_correct=False if hybrid_rejected else hybrid_hit,
        latency_ms=int(hybrid_time * 1000),
    ))
    
    print()

print("\n[4/4] 计算检索指标...")
print("=" * 70)

import re

def normalize_doc_name(name: str) -> str:
    """
    标准化文档名，去除年份、版本号、附件前缀等干扰信息
    
    Examples:
        "附件1：陕西省电力中长期市场实施细则（征求意见稿）" → "陕西电力中长期市场实施细则"
        "陕西电力中长期分时段交易实施细则（2025年10月修订版）" → "陕西电力中长期分时段交易实施细则"
    """
    if not name:
        return ""
    
    name = re.sub(r'附件\d*[：:]', '', name)
    name = re.sub(r'转载丨', '', name)
    name = re.sub(r'[《<>》]', '', name)
    name = re.sub(r'[（(][^）)]*[）)]', '', name)
    name = re.sub(r'\d{4}年\d*月?', '', name)
    name = re.sub(r'〔\d+〕', '', name)
    name = re.sub(r'（修订版|V\d+|征求意见稿|连续试运行|试运行）', '', name)
    name = re.sub(r'\s+', '', name)
    name = name.strip()
    
    return name

def check_hit(expected_docs, retrieved_docs):
    """
    检查expected_docs是否在retrieved_docs中（标准化后匹配）
    
    优化点：
    1. 去除年份、版本号等干扰信息
    2. 使用关键词匹配，只要核心关键词包含即可命中
    
    Examples:
        Expected: "陕西电力中长期市场实施细则"
        Retrieved: "陕西省电力中长期市场实施细则（征求意见稿）" -> True
        Expected: "陕西电力中长期分时段交易实施细则"
        Retrieved: "陕西电力市场中长期分时段交易实施细则" -> True (关键词匹配)
    """
    if not expected_docs or not retrieved_docs:
        return False
    
    for exp in expected_docs:
        exp_norm = normalize_doc_name(exp)
        for ret in retrieved_docs:
            ret_norm = normalize_doc_name(ret)
            if exp_norm and ret_norm:
                if exp_norm in ret_norm or ret_norm in exp_norm:
                    return True
                keywords = extract_keywords(exp_norm)
                if keywords and all(kw in ret_norm for kw in keywords):
                    return True
    return False

def extract_keywords(doc_name: str) -> List[str]:
    """
    从文档名中提取核心关键词
    
    Examples:
        "陕西电力中长期分时段交易实施细则" -> ["陕西", "电力", "中长期", "分时段", "交易", "实施细则"]
    """
    if not doc_name:
        return []
    
    keywords = []
    patterns = [
        r'陕西|陕西',
        r'电力',
        r'中长期|中长期',
        r'现货|现货',
        r'分时段|分时段',
        r'零售|零售',
        r'交易|交易',
        r'结算|结算',
        r'实施细则|实施细则',
        r'交易细则|交易细则',
        r'调频|调频',
        r'辅助服务|辅助',
        r'新型储能|储能',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, doc_name)
        if match:
            keywords.append(match.group())
    
    return keywords if len(keywords) >= 3 else []

for i, r in enumerate(baseline_results):
    r.is_correct = check_hit(test_configs[i].expected_docs, r.retrieved_doc_ids)

for i, r in enumerate(hybrid_results):
    r.is_correct = check_hit(test_configs[i].expected_docs, r.retrieved_doc_ids)

def custom_recall_at_k(results, k):
    hits = sum(1 for r in results if check_hit(r.expected_docs, r.retrieved_doc_ids[:k]))
    return hits / len(results) if results else 0.0

def custom_hit_rate(results):
    hits = sum(1 for r in results if r.is_correct)
    return hits / len(results) if results else 0.0

def compute_rejection_metrics(results, configs):
    """计算rejection相关指标"""
    correct_rejections = 0
    total_should_reject = 0
    incorrect_rejections = 0
    total_should_not_reject = 0
    
    for i, r in enumerate(results):
        cfg = configs[i]
        is_rejected = len(r.retrieved_doc_ids) == 0
        
        if cfg.should_reject:
            total_should_reject += 1
            if is_rejected:
                correct_rejections += 1
        else:
            total_should_not_reject += 1
            if is_rejected:
                incorrect_rejections += 1
    
    rejection_accuracy = correct_rejections / total_should_reject if total_should_reject > 0 else 0.0
    false_rejection_rate = incorrect_rejections / total_should_not_reject if total_should_not_reject > 0 else 0.0
    
    return {
        "rejection_accuracy": rejection_accuracy,
        "false_rejection_rate": false_rejection_rate,
        "correct_rejections": correct_rejections,
        "total_should_reject": total_should_reject,
        "incorrect_rejections": incorrect_rejections,
    }

baseline_metrics = compute_all_metrics(baseline_results)
hybrid_metrics = compute_all_metrics(hybrid_results)

baseline_recall3 = custom_recall_at_k(baseline_results, 3)
baseline_recall5 = custom_recall_at_k(baseline_results, 5)
baseline_hit_rate = custom_hit_rate(baseline_results)

hybrid_recall3 = custom_recall_at_k(hybrid_results, 3)
hybrid_recall5 = custom_recall_at_k(hybrid_results, 5)
hybrid_hit_rate = custom_hit_rate(hybrid_results)

baseline_avg_score = baseline_metrics.avg_score
hybrid_avg_score = hybrid_metrics.avg_score

baseline_calibrated_avg = calibrate_baseline_score(baseline_avg_score)
hybrid_calibrated_avg = calibrate_hybrid_score(hybrid_avg_score)

baseline_rejection_metrics = compute_rejection_metrics(baseline_results, test_configs)
hybrid_rejection_metrics = compute_rejection_metrics(hybrid_results, test_configs)

print("\n" + "=" * 70)
print("分数校准说明:")
print("=" * 70)
print(f"Baseline原始分数范围: {BASELINE_SCORE_MIN}-{BASELINE_SCORE_MAX}")
print(f"Hybrid原始分数范围:   {HYBRID_SCORE_MIN}-{HYBRID_SCORE_MAX}")
print("校准公式: calibrated = (score - min) / (max - min)")
print("=" * 70)

print("\nBaseline检索指标:")
print(f"  recall@3:  {baseline_recall3:.3f}")
print(f"  recall@5:  {baseline_recall5:.3f}")
print(f"  precision: {baseline_metrics.precision_at_k:.3f}")
print(f"  hit_rate:  {baseline_hit_rate:.3f}")
print(f"  avg_score (原始): {baseline_avg_score:.3f}")
print(f"  avg_score (校准): {baseline_calibrated_avg:.3f}")
print(f"  avg_latency: {baseline_metrics.avg_latency_ms:.0f}ms")
print(f"  rejection_accuracy: {baseline_rejection_metrics['rejection_accuracy']:.3f} ({baseline_rejection_metrics['correct_rejections']}/{baseline_rejection_metrics['total_should_reject']})")

print("\nHybrid检索指标:")
print(f"  recall@3:  {hybrid_recall3:.3f}")
print(f"  recall@5:  {hybrid_recall5:.3f}")
print(f"  precision: {hybrid_metrics.precision_at_k:.3f}")
print(f"  hit_rate:  {hybrid_hit_rate:.3f}")
print(f"  avg_score (原始): {hybrid_avg_score:.3f}")
print(f"  avg_score (校准): {hybrid_calibrated_avg:.3f}")
print(f"  avg_latency: {hybrid_metrics.avg_latency_ms:.0f}ms")
print(f"  rejection_accuracy: {hybrid_rejection_metrics['rejection_accuracy']:.3f} ({hybrid_rejection_metrics['correct_rejections']}/{hybrid_rejection_metrics['total_should_reject']})")

print("\n提升对比 (基于校准分数):")
print(f"  recall@3:  {((hybrid_recall3 - baseline_recall3) / max(baseline_recall3, 0.01)) * 100:+.1f}%")
print(f"  recall@5:  {((hybrid_recall5 - baseline_recall5) / max(baseline_recall5, 0.01)) * 100:+.1f}%")
print(f"  hit_rate:  {((hybrid_hit_rate - baseline_hit_rate) / max(baseline_hit_rate, 0.01)) * 100:+.1f}%")
print(f"  avg_score (校准): {((hybrid_calibrated_avg - baseline_calibrated_avg) / max(baseline_calibrated_avg, 0.01)) * 100:+.1f}%")

print("\nRejection对比:")
print(f"  Baseline: 正确拒绝 {baseline_rejection_metrics['correct_rejections']}/{baseline_rejection_metrics['total_should_reject']}, 错误拒绝 {baseline_rejection_metrics['incorrect_rejections']}")
print(f"  Hybrid:   正确拒绝 {hybrid_rejection_metrics['correct_rejections']}/{hybrid_rejection_metrics['total_should_reject']}, 错误拒绝 {hybrid_rejection_metrics['incorrect_rejections']}")

# 保存报告
report = {
    "experiment": "retrieval_only_shaaxi",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "calibration_config": {
        "baseline_range": f"{BASELINE_SCORE_MIN}-{BASELINE_SCORE_MAX}",
        "hybrid_range": f"{HYBRID_SCORE_MIN}-{HYBRID_SCORE_MAX}",
        "baseline_rejection_threshold": BASELINE_REJECTION_THRESHOLD,
        "hybrid_rejection_threshold": HYBRID_REJECTION_THRESHOLD,
    },
    "config": {
        "province": "陕西省",
        "test_size": len(test_configs),
        "bm25_k1": 1.5,
        "bm25_b": 0.6,
        "reranker": "BAAI/bge-reranker-base",
        "reranker_preload": preload_time,
        "score_normalization": True,
    },
    "baseline": {
        "recall_at_3": baseline_recall3,
        "recall_at_5": baseline_recall5,
        "precision_at_k": baseline_metrics.precision_at_k,
        "hit_rate": baseline_hit_rate,
        "avg_score_raw": baseline_avg_score,
        "avg_score_calibrated": baseline_calibrated_avg,
        "avg_latency_ms": baseline_metrics.avg_latency_ms,
        "rejection_accuracy": baseline_rejection_metrics["rejection_accuracy"],
        "correct_rejections": baseline_rejection_metrics["correct_rejections"],
        "total_should_reject": baseline_rejection_metrics["total_should_reject"],
    },
    "hybrid": {
        "recall_at_3": hybrid_recall3,
        "recall_at_5": hybrid_recall5,
        "precision_at_k": hybrid_metrics.precision_at_k,
        "hit_rate": hybrid_hit_rate,
        "avg_score_raw": hybrid_avg_score,
        "avg_score_calibrated": hybrid_calibrated_avg,
        "avg_latency_ms": hybrid_metrics.avg_latency_ms,
        "rejection_accuracy": hybrid_rejection_metrics["rejection_accuracy"],
        "correct_rejections": hybrid_rejection_metrics["correct_rejections"],
        "total_should_reject": hybrid_rejection_metrics["total_should_reject"],
    },
    "improvement": {
        "recall_at_3": ((hybrid_recall3 - baseline_recall3) / max(baseline_recall3, 0.01)) * 100,
        "recall_at_5": ((hybrid_recall5 - baseline_recall5) / max(baseline_recall5, 0.01)) * 100,
        "hit_rate": ((hybrid_hit_rate - baseline_hit_rate) / max(baseline_hit_rate, 0.01)) * 100,
        "avg_score_calibrated": ((hybrid_calibrated_avg - baseline_calibrated_avg) / max(baseline_calibrated_avg, 0.01)) * 100,
    },
    "details": [
        {
            "question_id": cfg.question_id,
            "category": cfg.category,
            "expected_docs": cfg.expected_docs,
            "should_reject": cfg.should_reject,
            "baseline_hit": baseline_results[i].is_correct,
            "hybrid_hit": hybrid_results[i].is_correct,
            "baseline_rejected": len(baseline_results[i].retrieved_doc_ids) == 0,
            "hybrid_rejected": len(hybrid_results[i].retrieved_doc_ids) == 0,
            "baseline_docs": baseline_results[i].retrieved_doc_ids[:3],
            "hybrid_docs": hybrid_results[i].retrieved_doc_ids[:3],
        }
        for i, cfg in enumerate(test_configs)
    ],
}

output_file = Path("evaluation/reports_hybrid/retrieval_metrics_shaaxi.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n报告已保存: {output_file}")

print("\n" + "=" * 70)
print("简化评估完成!")
print("提示: 如需完整评估（包括Ragas生成指标），请运行:")
print("  python scripts/run_full_evaluation.py")
print("=" * 70)
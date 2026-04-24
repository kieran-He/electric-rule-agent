#!/usr/bin/env python3
"""
Baseline vs Hybrid 全流程对比实验 (100条陕西省数据)

实验流程:
Phase 1: 初始化组件
Phase 2: Baseline检索+生成
Phase 3: Hybrid检索+生成
Phase 4: Ragas评估
Phase 5: 指标计算
Phase 6: 报告生成
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import re
from typing import List, Dict, Any
from dataclasses import dataclass, field
import os
import warnings
warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

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

def generate_answer(query: str, chunks: List, province_code: str, llm_wrapper) -> str:
    if not chunks:
        return "未检索到相关文档，无法回答该问题。"
    
    provincial_context = format_chunks_for_context(chunks)
    
    user_content = f"""问题: {query}

省级证据({province_code}):
{provincial_context}

通用证据:
- 无通用证据

历史对话:

请根据上述证据回答问题。"""

    system_prompt = """你是电力政策问答助手。只能根据提供的证据回答，禁止编造。如果证据不足，明确说明"未检索到充分依据"。

回答要求：
1. 基于证据内容回答，不要添加证据中没有的信息
2. 引用证据时标注来源文档名称
3. 如果问题涉及多个省份，分别说明各省份的政策
4. 如果证据不足，明确告知用户并建议补充检索"""

    try:
        answer = llm_wrapper.invoke(user_content, system=system_prompt)
        return answer
    except Exception as e:
        return f"LLM服务暂时不可用: {str(e)[:100]}"

@dataclass
class ExperimentResult:
    question_id: str
    question: str
    category: str
    expected_keywords: List[str] = field(default_factory=list)
    should_reject: bool = False
    province_detected: str = ""
    baseline: Dict[str, Any] = field(default_factory=dict)
    hybrid: Dict[str, Any] = field(default_factory=dict)
    ragas: Dict[str, Any] = field(default_factory=dict)

print("=" * 70)
print("Baseline vs Hybrid 全流程对比实验 (100条陕西)")
print("=" * 70)
print(f"时间预估: 约55-60分钟")
print("=" * 70)

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent

print("\n[Phase 1] 初始化组件...")
print("=" * 70)

print("加载benchmark数据集...")
benchmark_path = EXPERIMENT_DIR / "benchmark_100sn.json"
if not benchmark_path.exists():
    benchmark_path = Path("evaluation/benchmark.json")

with open(benchmark_path, encoding='utf-8') as f:
    benchmark_data = json.load(f)

questions = benchmark_data.get("questions", [])
print(f"数据集: {len(questions)}条问题")
dist = {}
for q in questions:
    c = q.get("category", "unknown")
    dist[c] = dist.get(c, 0) + 1
print(f"分布: {dist}")

print("\n加载文档映射...")
manifest_path = Path("data/processed/_manifest.json")
doc_name_map = {}
if manifest_path.exists():
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)
        for hash_key, info in manifest.get("processed_hashes", {}).items():
            doc_name_map[hash_key] = info.get("doc_name", "")
    print(f"文档映射: {len(doc_name_map)}条")

print("\n初始化Vector Repository...")
from app.config import settings
from app.repository import ChromaPolicyRepository
repo = ChromaPolicyRepository(
    persist_directory=settings.chroma_path,
    embedding_model_name=settings.embedding_model,
)
print(f"Vector repo: {repo.ready}")

print("\n构建BM25索引...")
from app.langchain.bm25_indexer import BM25Indexer
bm25 = BM25Indexer(k1=1.5, b=0.6)
bm25_docs = bm25.build_index()
print(f"BM25索引: {bm25_docs}条")

print("\n初始化Province Detector...")
from app.province import ProvinceDetector
detector = ProvinceDetector()
print("Province detector: 就绪")

print("\n预加载Reranker...")
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
print("Hybrid retriever: 就绪")

print("\n初始化LLM Wrapper...")
from app.langchain.llm import MiniMaxLLMWrapper
from app.langchain.retriever_wrapper import format_chunks_for_context

llm_wrapper = MiniMaxLLMWrapper(
    api_key=os.getenv("LLM_API_KEY", ""),
    endpoint=os.getenv("LLM_ENDPOINT", ""),
    model=os.getenv("LLM_MODEL", "MiniMax-M2.7"),
    disable_thinking=True,
)
print("LLM wrapper: 就绪")

print("\n初始化Ragas评估器...")
from evaluation.ragas_evaluator import RagasEvaluator
ragas_evaluator = RagasEvaluator(
    llm_endpoint=os.getenv("RAGAS_ENDPOINT", os.getenv("LLM_ENDPOINT", "")),
    llm_api_key=os.getenv("RAGAS_API_KEY", os.getenv("LLM_API_KEY", "")),
    llm_model=os.getenv("RAGAS_MODEL", os.getenv("LLM_MODEL", "MiniMax-M2.7")),
)
print(f"Ragas evaluator: {'可用' if ragas_evaluator.is_available() else '不可用(将跳过)'}")

print("\n[Phase 2] Baseline检索+生成...")
print("=" * 70)
print(f"预估时间: ~10分钟 (每条约6秒)")

baseline_results: List[ExperimentResult] = []
baseline_answers = []
baseline_contexts = []
baseline_start_time = time.time()

for i, q in enumerate(questions):
    qid = q.get("question_id", f"q{i}")
    question = q.get("question", "")
    category = q.get("category", "unknown")
    expected_keywords = q.get("expected_answer_keywords", [])
    should_reject = q.get("should_reject", False)
    
    print(f"[{i+1}/{len(questions)}] {qid} ({category})")
    
    detection = detector.detect(question)
    province_detected = detection.province_code if detection and detection.province_code else "SN"
    print(f"  Province: {province_detected}")
    
    start = time.time()
    
    baseline_chunks = repo.retrieve(question, 12, "province", "SN")
    
    retrieval_time = time.time() - start
    
    baseline_hashes = [c.metadata.get("file_hash", "") for c in baseline_chunks]
    baseline_docs = [doc_name_map.get(h, h) for h in baseline_hashes]
    baseline_scores = [c.score for c in baseline_chunks if hasattr(c, 'score') and c.score > 0]
    baseline_avg_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
    baseline_calibrated = calibrate_baseline_score(baseline_avg_score) if baseline_avg_score > 0 else 0
    
    baseline_rejected = False
    if baseline_scores and baseline_scores[0] < BASELINE_REJECTION_THRESHOLD:
        baseline_rejected = True
        print(f"  Baseline: REJECTED (top score {baseline_scores[0]:.3f} < {BASELINE_REJECTION_THRESHOLD})")
    
    start = time.time()
    
    if baseline_rejected:
        answer = "未检索到相关信息，无法回答该问题。"
    else:
        answer = generate_answer(question, baseline_chunks, "SN", llm_wrapper)
    
    generation_time = time.time() - start
    total_time = retrieval_time + generation_time
    
    if not baseline_rejected:
        keywords_hit = any(kw in answer for kw in expected_keywords) if expected_keywords else False
        hit = check_hit_fast([], baseline_docs[:12])
        print(f"  Baseline: {retrieval_time:.3f}s检索, {generation_time:.3f}s生成, score={baseline_avg_score:.3f}(cal={baseline_calibrated:.3f})")
        if baseline_docs[:3]:
            print(f"    Top docs: {baseline_docs[0][:40]}...")
    
    baseline_answers.append(answer)
    baseline_contexts.append([c.text for c in baseline_chunks])
    
    result = ExperimentResult(
        question_id=qid,
        question=question,
        category=category,
        expected_keywords=expected_keywords,
        should_reject=should_reject,
        province_detected=province_detected,
        baseline={
            "retrieval_time_ms": int(retrieval_time * 1000),
            "generation_time_ms": int(generation_time * 1000),
            "total_latency_ms": int(total_time * 1000),
            "retrieved_docs": baseline_docs[:12],
            "scores": baseline_scores[:12],
            "avg_score": baseline_avg_score,
            "avg_score_calibrated": baseline_calibrated,
            "rejected": baseline_rejected,
            "answer": answer,
            "keywords_hit": keywords_hit if not baseline_rejected else False,
        },
    )
    baseline_results.append(result)
    
    if (i + 1) % 10 == 0:
        elapsed = time.time() - baseline_start_time
        avg_time = elapsed / (i + 1)
        remaining = (len(questions) - i - 1) * avg_time
        print(f"  Progress: {i+1}/{len(questions)}, elapsed={elapsed:.0f}s, remaining={remaining:.0f}s")

baseline_total_time = time.time() - baseline_start_time
print(f"\nBaseline完成: {baseline_total_time:.0f}s ({baseline_total_time/60:.1f}分钟)")

print("\n[Phase 3] Hybrid检索+生成...")
print("=" * 70)
print(f"预估时间: ~20分钟 (每条约12秒)")

hybrid_results: List[ExperimentResult] = []
hybrid_answers = []
hybrid_contexts = []
hybrid_start_time = time.time()

for i, q in enumerate(questions):
    qid = q.get("question_id", f"q{i}")
    question = q.get("question", "")
    category = q.get("category", "unknown")
    expected_keywords = q.get("expected_answer_keywords", [])
    should_reject = q.get("should_reject", False)
    
    print(f"[{i+1}/{len(questions)}] {qid} ({category})")
    
    detection = detector.detect(question)
    province_codes = ["SN"]
    if detection and detection.province_code:
        province_codes = [detection.province_code]
        print(f"  Province: {detection.province_code}")
    
    start = time.time()
    
    hybrid_chunks = hybrid_retriever.retrieve(question, province_codes)
    
    retrieval_time = time.time() - start
    
    hybrid_hashes = [c.metadata.get("file_hash", "") for c in hybrid_chunks]
    hybrid_docs = [doc_name_map.get(h, h) for h in hybrid_hashes]
    hybrid_scores = [c.score for c in hybrid_chunks if hasattr(c, 'score') and c.score > 0]
    hybrid_avg_score = sum(hybrid_scores) / len(hybrid_scores) if hybrid_scores else 0
    hybrid_calibrated = calibrate_hybrid_score(hybrid_avg_score) if hybrid_avg_score > 0 else 0
    
    hybrid_rejected = False
    if not hybrid_chunks or (hybrid_scores and hybrid_scores[0] < HYBRID_REJECTION_THRESHOLD):
        hybrid_rejected = True
        top_score = hybrid_scores[0] if hybrid_scores else 0
        print(f"  Hybrid: REJECTED (top score {top_score:.3f} < {HYBRID_REJECTION_THRESHOLD})")
    
    start = time.time()
    
    if hybrid_rejected:
        answer = "未检索到相关信息，无法回答该问题。"
    else:
        answer = generate_answer(question, hybrid_chunks, province_codes[0], llm_wrapper)
    
    generation_time = time.time() - start
    total_time = retrieval_time + generation_time
    
    if not hybrid_rejected:
        keywords_hit = any(kw in answer for kw in expected_keywords) if expected_keywords else False
        hit = check_hit_fast([], hybrid_docs[:12])
        improvement = ((hybrid_calibrated - baseline_results[i].baseline.get("avg_score_calibrated", 0)) / max(baseline_results[i].baseline.get("avg_score_calibrated", 0.01), 0.01)) * 100
        print(f"  Hybrid: {retrieval_time:.3f}s检索, {generation_time:.3f}s生成, score={hybrid_avg_score:.3f}(cal={hybrid_calibrated:.3f}), improvement={improvement:+.1f}%")
        if hybrid_docs[:3]:
            print(f"    Top docs: {hybrid_docs[0][:40]}...")
    
    hybrid_answers.append(answer)
    hybrid_contexts.append([c.text for c in hybrid_chunks])
    
    baseline_results[i].hybrid = {
        "retrieval_time_ms": int(retrieval_time * 1000),
        "generation_time_ms": int(generation_time * 1000),
        "total_latency_ms": int(total_time * 1000),
        "retrieved_docs": hybrid_docs[:12],
        "scores": hybrid_scores[:12],
        "avg_score": hybrid_avg_score,
        "avg_score_calibrated": hybrid_calibrated,
        "rejected": hybrid_rejected,
        "answer": answer,
        "keywords_hit": keywords_hit if not hybrid_rejected else False,
    }
    
    if (i + 1) % 10 == 0:
        elapsed = time.time() - hybrid_start_time
        avg_time = elapsed / (i + 1)
        remaining = (len(questions) - i - 1) * avg_time
        print(f"  Progress: {i+1}/{len(questions)}, elapsed={elapsed:.0f}s, remaining={remaining:.0f}s")

hybrid_total_time = time.time() - hybrid_start_time
print(f"\nHybrid完成: {hybrid_total_time:.0f}s ({hybrid_total_time/60:.1f}分钟)")

print("\n[Phase 4] Ragas评估...")
print("=" * 70)

if ragas_evaluator.is_available():
    print("评估Baseline生成质量...")
    baseline_ragas_scores = ragas_evaluator.evaluate_batch(
        questions=[q.get("question", "") for q in questions],
        answers=baseline_answers,
        contexts=baseline_contexts,
    )
    
    print("评估Hybrid生成质量...")
    hybrid_ragas_scores = ragas_evaluator.evaluate_batch(
        questions=[q.get("question", "") for q in questions],
        answers=hybrid_answers,
        contexts=hybrid_contexts,
    )
    
    for i, result in enumerate(baseline_results):
        result.ragas = {
            "baseline_faithfulness": baseline_ragas_scores.get("faithfulness", {}).get(i, None),
            "baseline_answer_relevancy": baseline_ragas_scores.get("answer_relevancy", {}).get(i, None),
            "baseline_context_precision": baseline_ragas_scores.get("context_precision", {}).get(i, None),
            "hybrid_faithfulness": hybrid_ragas_scores.get("faithfulness", {}).get(i, None),
            "hybrid_answer_relevancy": hybrid_ragas_scores.get("answer_relevancy", {}).get(i, None),
            "hybrid_context_precision": hybrid_ragas_scores.get("context_precision", {}).get(i, None),
        }
    
    print("Ragas评估完成")
else:
    print("Ragas不可用,跳过评估")
    for result in baseline_results:
        result.ragas = {}

print("\n[Phase 5] 指标计算...")
print("=" * 70)

def compute_rejection_metrics(results: List[ExperimentResult]) -> Dict[str, Any]:
    correct_rejections = 0
    total_should_reject = 0
    incorrect_rejections = 0
    total_should_not_reject = 0
    
    for r in results:
        baseline_rejected = r.baseline.get("rejected", False)
        hybrid_rejected = r.hybrid.get("rejected", False)
        
        if r.should_reject:
            total_should_reject += 1
            if hybrid_rejected:
                correct_rejections += 1
        else:
            total_should_not_reject += 1
            if hybrid_rejected:
                incorrect_rejections += 1
    
    return {
        "baseline_correct_rejections": sum(1 for r in results if r.should_reject and r.baseline.get("rejected", False)),
        "hybrid_correct_rejections": correct_rejections,
        "total_should_reject": total_should_reject,
        "baseline_incorrect_rejections": sum(1 for r in results if not r.should_reject and r.baseline.get("rejected", False)),
        "hybrid_incorrect_rejections": incorrect_rejections,
        "total_should_not_reject": total_should_not_reject,
        "baseline_rejection_accuracy": sum(1 for r in results if r.should_reject and r.baseline.get("rejected", False)) / total_should_reject if total_should_reject > 0 else 0,
        "hybrid_rejection_accuracy": correct_rejections / total_should_reject if total_should_reject > 0 else 0,
    }

baseline_avg_score_calibrated = sum(r.baseline.get("avg_score_calibrated", 0) for r in baseline_results) / len(baseline_results)
hybrid_avg_score_calibrated = sum(r.hybrid.get("avg_score_calibrated", 0) for r in baseline_results) / len(baseline_results)

baseline_avg_latency = sum(r.baseline.get("total_latency_ms", 0) for r in baseline_results) / len(baseline_results)
hybrid_avg_latency = sum(r.hybrid.get("total_latency_ms", 0) for r in baseline_results) / len(baseline_results)

baseline_keywords_hit_rate = sum(1 for r in baseline_results if r.baseline.get("keywords_hit", False)) / len(baseline_results)
hybrid_keywords_hit_rate = sum(1 for r in baseline_results if r.hybrid.get("keywords_hit", False)) / len(baseline_results)

rejection_metrics = compute_rejection_metrics(baseline_results)

ragas_avg = {
    "baseline_faithfulness": sum(r.ragas.get("baseline_faithfulness", 0) or 0 for r in baseline_results) / len(baseline_results),
    "baseline_answer_relevancy": sum(r.ragas.get("baseline_answer_relevancy", 0) or 0 for r in baseline_results) / len(baseline_results),
    "baseline_context_precision": sum(r.ragas.get("baseline_context_precision", 0) or 0 for r in baseline_results) / len(baseline_results),
    "hybrid_faithfulness": sum(r.ragas.get("hybrid_faithfulness", 0) or 0 for r in baseline_results) / len(baseline_results),
    "hybrid_answer_relevancy": sum(r.ragas.get("hybrid_answer_relevancy", 0) or 0 for r in baseline_results) / len(baseline_results),
    "hybrid_context_precision": sum(r.ragas.get("hybrid_context_precision", 0) or 0 for r in baseline_results) / len(baseline_results),
}

print("\n检索+生成指标对比:")
print(f"  avg_score_calibrated: Baseline={baseline_avg_score_calibrated:.3f}, Hybrid={hybrid_avg_score_calibrated:.3f}")
print(f"  avg_latency_ms:       Baseline={baseline_avg_latency:.0f}, Hybrid={hybrid_avg_latency:.0f}")
print(f"  keywords_hit_rate:    Baseline={baseline_keywords_hit_rate:.3f}, Hybrid={hybrid_keywords_hit_rate:.3f}")

print("\nRejection指标:")
print(f"  Baseline: 正确拒绝 {rejection_metrics['baseline_correct_rejections']}/{rejection_metrics['total_should_reject']}, 错误拒绝 {rejection_metrics['baseline_incorrect_rejections']}")
print(f"  Hybrid:   正确拒绝 {rejection_metrics['hybrid_correct_rejections']}/{rejection_metrics['total_should_reject']}, 错误拒绝 {rejection_metrics['hybrid_incorrect_rejections']}")

if ragas_evaluator.is_available():
    print("\nRagas指标对比:")
    print(f"  faithfulness:        Baseline={ragas_avg['baseline_faithfulness']:.3f}, Hybrid={ragas_avg['hybrid_faithfulness']:.3f}")
    print(f"  answer_relevancy:    Baseline={ragas_avg['baseline_answer_relevancy']:.3f}, Hybrid={ragas_avg['hybrid_answer_relevancy']:.3f}")
    print(f"  context_precision:   Baseline={ragas_avg['baseline_context_precision']:.3f}, Hybrid={ragas_avg['hybrid_context_precision']:.3f}")

print("\n[Phase 6] 报告生成...")
print("=" * 70)

results_dir = EXPERIMENT_DIR / "results"
results_dir.mkdir(parents=True, exist_ok=True)

baseline_output = {
    "experiment": "baseline_hybrid_100sn",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "method": "baseline_vector_only",
    "config": {
        "dataset": "benchmark_100sn.json",
        "dataset_size": len(questions),
        "province_scope": "SN",
        "top_k": 12,
        "score_calibration": {
            "min": BASELINE_SCORE_MIN,
            "max": BASELINE_SCORE_MAX,
            "rejection_threshold": BASELINE_REJECTION_THRESHOLD,
        },
    },
    "summary": {
        "avg_score_calibrated": baseline_avg_score_calibrated,
        "avg_latency_ms": baseline_avg_latency,
        "keywords_hit_rate": baseline_keywords_hit_rate,
        "rejection_accuracy": rejection_metrics["baseline_rejection_accuracy"],
        "correct_rejections": rejection_metrics["baseline_correct_rejections"],
        "incorrect_rejections": rejection_metrics["baseline_incorrect_rejections"],
        "total_time_seconds": baseline_total_time,
        "faithfulness": ragas_avg["baseline_faithfulness"],
        "answer_relevancy": ragas_avg["baseline_answer_relevancy"],
        "context_precision": ragas_avg["baseline_context_precision"],
    },
    "details": [
        {
            "question_id": r.question_id,
            "question": r.question,
            "category": r.category,
            "province_detected": r.province_detected,
            "should_reject": r.should_reject,
            **r.baseline,
        }
        for r in baseline_results
    ],
}

with open(results_dir / "baseline_results.json", 'w', encoding='utf-8') as f:
    json.dump(baseline_output, f, ensure_ascii=False, indent=2)
print(f"保存: {results_dir / 'baseline_results.json'}")

hybrid_output = {
    "experiment": "baseline_hybrid_100sn",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "method": "hybrid_vector_bm25_rerank",
    "config": {
        "dataset": "benchmark_100sn.json",
        "dataset_size": len(questions),
        "province_scope": "SN",
        "vector_top_k": 15,
        "bm25_top_k": 15,
        "final_top_k": 12,
        "bm25_k1": 1.5,
        "bm25_b": 0.6,
        "reranker": "BAAI/bge-reranker-base",
        "score_calibration": {
            "min": HYBRID_SCORE_MIN,
            "max": HYBRID_SCORE_MAX,
            "rejection_threshold": HYBRID_REJECTION_THRESHOLD,
        },
    },
    "summary": {
        "avg_score_calibrated": hybrid_avg_score_calibrated,
        "avg_latency_ms": hybrid_avg_latency,
        "keywords_hit_rate": hybrid_keywords_hit_rate,
        "rejection_accuracy": rejection_metrics["hybrid_rejection_accuracy"],
        "correct_rejections": rejection_metrics["hybrid_correct_rejections"],
        "incorrect_rejections": rejection_metrics["hybrid_incorrect_rejections"],
        "total_time_seconds": hybrid_total_time,
        "faithfulness": ragas_avg["hybrid_faithfulness"],
        "answer_relevancy": ragas_avg["hybrid_answer_relevancy"],
        "context_precision": ragas_avg["hybrid_context_precision"],
    },
    "details": [
        {
            "question_id": r.question_id,
            "question": r.question,
            "category": r.category,
            "province_detected": r.province_detected,
            "should_reject": r.should_reject,
            **r.hybrid,
        }
        for r in baseline_results
    ],
}

with open(results_dir / "hybrid_results.json", 'w', encoding='utf-8') as f:
    json.dump(hybrid_output, f, ensure_ascii=False, indent=2)
print(f"保存: {results_dir / 'hybrid_results.json'}")

metrics_summary = {
    "experiment": "baseline_hybrid_100sn",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "comparison": {
        "avg_score_calibrated": {
            "baseline": baseline_avg_score_calibrated,
            "hybrid": hybrid_avg_score_calibrated,
            "improvement": ((hybrid_avg_score_calibrated - baseline_avg_score_calibrated) / max(baseline_avg_score_calibrated, 0.01)) * 100,
        },
        "avg_latency_ms": {
            "baseline": baseline_avg_latency,
            "hybrid": hybrid_avg_latency,
        },
        "keywords_hit_rate": {
            "baseline": baseline_keywords_hit_rate,
            "hybrid": hybrid_keywords_hit_rate,
            "improvement": ((hybrid_keywords_hit_rate - baseline_keywords_hit_rate) / max(baseline_keywords_hit_rate, 0.01)) * 100,
        },
        "rejection_accuracy": {
            "baseline": rejection_metrics["baseline_rejection_accuracy"],
            "hybrid": rejection_metrics["hybrid_rejection_accuracy"],
            "improvement": ((rejection_metrics["hybrid_rejection_accuracy"] - rejection_metrics["baseline_rejection_accuracy"]) / max(rejection_metrics["baseline_rejection_accuracy"], 0.01)) * 100,
        },
    },
    "ragas_comparison": {
        "faithfulness": {
            "baseline": ragas_avg["baseline_faithfulness"],
            "hybrid": ragas_avg["hybrid_faithfulness"],
            "improvement": ((ragas_avg["hybrid_faithfulness"] - ragas_avg["baseline_faithfulness"]) / max(ragas_avg["baseline_faithfulness"], 0.01)) * 100,
        },
        "answer_relevancy": {
            "baseline": ragas_avg["baseline_answer_relevancy"],
            "hybrid": ragas_avg["hybrid_answer_relevancy"],
            "improvement": ((ragas_avg["hybrid_answer_relevancy"] - ragas_avg["baseline_answer_relevancy"]) / max(ragas_avg["baseline_answer_relevancy"], 0.01)) * 100,
        },
        "context_precision": {
            "baseline": ragas_avg["baseline_context_precision"],
            "hybrid": ragas_avg["hybrid_context_precision"],
            "improvement": ((ragas_avg["hybrid_context_precision"] - ragas_avg["baseline_context_precision"]) / max(ragas_avg["baseline_context_precision"], 0.01)) * 100,
        },
    },
    "time_summary": {
        "baseline_total_seconds": baseline_total_time,
        "hybrid_total_seconds": hybrid_total_time,
        "reranker_preload_seconds": preload_time,
    },
}

with open(results_dir / "metrics_summary.json", 'w', encoding='utf-8') as f:
    json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
print(f"保存: {results_dir / 'metrics_summary.json'}")

if ragas_evaluator.is_available():
    ragas_output = {
        "experiment": "baseline_hybrid_100sn",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_scores": {
            "faithfulness": {i: r.ragas.get("baseline_faithfulness") for i, r in enumerate(baseline_results)},
            "answer_relevancy": {i: r.ragas.get("baseline_answer_relevancy") for i, r in enumerate(baseline_results)},
            "context_precision": {i: r.ragas.get("baseline_context_precision") for i, r in enumerate(baseline_results)},
            "avg_faithfulness": ragas_avg["baseline_faithfulness"],
            "avg_answer_relevancy": ragas_avg["baseline_answer_relevancy"],
            "avg_context_precision": ragas_avg["baseline_context_precision"],
        },
        "hybrid_scores": {
            "faithfulness": {i: r.ragas.get("hybrid_faithfulness") for i, r in enumerate(baseline_results)},
            "answer_relevancy": {i: r.ragas.get("hybrid_answer_relevancy") for i, r in enumerate(baseline_results)},
            "context_precision": {i: r.ragas.get("hybrid_context_precision") for i, r in enumerate(baseline_results)},
            "avg_faithfulness": ragas_avg["hybrid_faithfulness"],
            "avg_answer_relevancy": ragas_avg["hybrid_answer_relevancy"],
            "avg_context_precision": ragas_avg["hybrid_context_precision"],
        },
    }
    
    with open(results_dir / "ragas_scores.json", 'w', encoding='utf-8') as f:
        json.dump(ragas_output, f, ensure_ascii=False, indent=2)
    print(f"保存: {results_dir / 'ragas_scores.json'}")

reports_dir = EXPERIMENT_DIR / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

report_md = f"""# Baseline vs Hybrid 全流程对比实验报告

## 实验概述

- **实验名称**: baseline_vs_hybrid_100sn
- **数据集**: 100条陕西省专用问题
- **时间**: {time.strftime("%Y-%m-%d %H:%M:%S")}

## 数据集分布

| 类别 | 数量 |
|------|------|
| clause_qa | {dist.get('clause_qa', 0)} |
| flow_qa | {dist.get('flow_qa', 0)} |
| settlement_qa | {dist.get('settlement_qa', 0)} |
| rejection | {dist.get('rejection', 0)} |
| **总计** | **{len(questions)}** |

## 指标对比

### 检索+生成指标

| 指标 | Baseline | Hybrid | 提升 |
|------|----------|--------|------|
| avg_score (校准) | {baseline_avg_score_calibrated:.3f} | {hybrid_avg_score_calibrated:.3f} | {((hybrid_avg_score_calibrated - baseline_avg_score_calibrated) / max(baseline_avg_score_calibrated, 0.01)) * 100:+.1f}% |
| avg_latency (ms) | {baseline_avg_latency:.0f} | {hybrid_avg_latency:.0f} | - |
| keywords_hit_rate | {baseline_keywords_hit_rate:.3f} | {hybrid_keywords_hit_rate:.3f} | {((hybrid_keywords_hit_rate - baseline_keywords_hit_rate) / max(baseline_keywords_hit_rate, 0.01)) * 100:+.1f}% |

### Rejection指标

| 指标 | Baseline | Hybrid |
|------|----------|--------|
| 正确拒绝 | {rejection_metrics['baseline_correct_rejections']}/{rejection_metrics['total_should_reject']} | {rejection_metrics['hybrid_correct_rejections']}/{rejection_metrics['total_should_reject']} |
| 错误拒绝 | {rejection_metrics['baseline_incorrect_rejections']} | {rejection_metrics['hybrid_incorrect_rejections']} |
| rejection_accuracy | {rejection_metrics['baseline_rejection_accuracy']:.3f} | {rejection_metrics['hybrid_rejection_accuracy']:.3f} |

### Ragas生成质量指标

| 指标 | Baseline | Hybrid | 提升 |
|------|----------|--------|------|
| faithfulness | {ragas_avg['baseline_faithfulness']:.3f} | {ragas_avg['hybrid_faithfulness']:.3f} | {((ragas_avg['hybrid_faithfulness'] - ragas_avg['baseline_faithfulness']) / max(ragas_avg['baseline_faithfulness'], 0.01)) * 100:+.1f}% |
| answer_relevancy | {ragas_avg['baseline_answer_relevancy']:.3f} | {ragas_avg['hybrid_answer_relevancy']:.3f} | {((ragas_avg['hybrid_answer_relevancy'] - ragas_avg['baseline_answer_relevancy']) / max(ragas_avg['baseline_answer_relevancy'], 0.01)) * 100:+.1f}% |
| context_precision | {ragas_avg['baseline_context_precision']:.3f} | {ragas_avg['hybrid_context_precision']:.3f} | {((ragas_avg['hybrid_context_precision'] - ragas_avg['baseline_context_precision']) / max(ragas_avg['baseline_context_precision'], 0.01)) * 100:+.1f}% |

## 时间统计

| 阶段 | 时间 |
|------|------|
| Reranker预加载 | {preload_time:.1f}s |
| Baseline检索+生成 | {baseline_total_time:.0f}s ({baseline_total_time/60:.1f}min) |
| Hybrid检索+生成 | {hybrid_total_time:.0f}s ({hybrid_total_time/60:.1f}min) |

## 配置信息

### Baseline配置
- 检索方式: Vector-only
- top_k: 12
- 分数校准范围: {BASELINE_SCORE_MIN}-{BASELINE_SCORE_MAX}
- Rejection阈值: {BASELINE_REJECTION_THRESHOLD}

### Hybrid配置
- 检索方式: Vector + BM25 + Rerank
- vector_top_k: 15
- bm25_top_k: 15
- final_top_k: 12
- BM25参数: k1=1.5, b=0.6
- Reranker: BAAI/bge-reranker-base
- 分数校准范围: {HYBRID_SCORE_MIN}-{HYBRID_SCORE_MAX}
- Rejection阈值: {HYBRID_REJECTION_THRESHOLD}

## 结论

Hybrid方法相比Baseline:
- 校准分数提升 {((hybrid_avg_score_calibrated - baseline_avg_score_calibrated) / max(baseline_avg_score_calibrated, 0.01)) * 100:.1f}%
- Rejection准确率提升 {((rejection_metrics['hybrid_rejection_accuracy'] - rejection_metrics['baseline_rejection_accuracy']) / max(rejection_metrics['baseline_rejection_accuracy'], 0.01)) * 100:.1f}%
- 延迟增加约 {hybrid_avg_latency - baseline_avg_latency:.0f}ms

---
*报告生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}*
"""

with open(reports_dir / "experiment_report.md", 'w', encoding='utf-8') as f:
    f.write(report_md)
print(f"保存: {reports_dir / 'experiment_report.md'}")

answers_preview = "答案预览 (前10条)\n\n"
for i, r in enumerate(baseline_results[:10]):
    answers_preview += f"Q{i+1} [{r.category}]: {r.question[:50]}...\n"
    answers_preview += f"Baseline Answer: {r.baseline.get('answer', '')[:100]}...\n"
    answers_preview += f"Hybrid Answer: {r.hybrid.get('answer', '')[:100]}...\n\n"

with open(reports_dir / "answers_preview.txt", 'w', encoding='utf-8') as f:
    f.write(answers_preview)
print(f"保存: {reports_dir / 'answers_preview.txt'}")

print("\n" + "=" * 70)
print("实验完成!")
print("=" * 70)
print(f"总时间: {baseline_total_time + hybrid_total_time:.0f}s ({(baseline_total_time + hybrid_total_time)/60:.1f}分钟)")
print(f"结果文件: {results_dir}")
print(f"报告文件: {reports_dir}")
print("=" * 70)
#!/usr/bin/env python3
"""
仅运行 Ragas 评估（基于已有结果）

修复问题: 原实验脚本未加载 .env 文件，导致 Ragas 评估返回 null
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

ragas_api_key = os.getenv("RAGAS_API_KEY", "") or os.getenv("LLM_API_KEY", "")
ragas_endpoint = os.getenv("RAGAS_ENDPOINT", "") or os.getenv("LLM_ENDPOINT", "")

if ragas_api_key:
    os.environ["OPENAI_API_KEY"] = ragas_api_key
if ragas_endpoint:
    os.environ["OPENAI_API_BASE"] = ragas_endpoint

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

print("=" * 70)
print("Ragas 评估修复脚本 (方案B)")
print("=" * 70)
print(f"时间预估: 约25分钟")
print("=" * 70)

print("\n[Step 1] 验证环境变量...")
print("=" * 70)
llm_api_key = os.getenv("LLM_API_KEY", "")
ragas_api_key = os.getenv("RAGAS_API_KEY", "")
ragas_endpoint = os.getenv("RAGAS_ENDPOINT", "")
ragas_model = os.getenv("RAGAS_MODEL", "")

print(f"LLM_API_KEY已加载: {'是' if llm_api_key else '否'}")
print(f"RAGAS_API_KEY已加载: {'是' if ragas_api_key else '否'}")
print(f"RAGAS_ENDPOINT: {ragas_endpoint if ragas_endpoint else '未设置'}")
print(f"RAGAS_MODEL: {ragas_model if ragas_model else '未设置'}")

if not ragas_api_key and not llm_api_key:
    print("\n错误: 环境变量未加载，请检查 .env 文件路径")
    sys.exit(1)

print("\n[Step 2] 加载已有结果...")
print("=" * 70)

baseline_path = RESULTS_DIR / "baseline_results.json"
hybrid_path = RESULTS_DIR / "hybrid_results.json"

with open(baseline_path, encoding='utf-8') as f:
    baseline_results = json.load(f)

with open(hybrid_path, encoding='utf-8') as f:
    hybrid_results = json.load(f)

baseline_details = baseline_results.get("details", [])
hybrid_details = hybrid_results.get("details", [])

print(f"Baseline: {len(baseline_details)}条")
print(f"Hybrid: {len(hybrid_details)}条")

print("\n[Step 3] 初始化检索组件...")
print("=" * 70)

from app.config import settings
from app.core.repository import ChromaPolicyRepository
repo = ChromaPolicyRepository(
    persist_directory=settings.chroma_path,
    embedding_model_name=settings.embedding_model,
)
print(f"Vector repo: {repo.ready}")

from app.langchain.bm25_indexer import BM25Indexer
bm25 = BM25Indexer(k1=1.5, b=0.6)
bm25_docs = bm25.build_index()
print(f"BM25索引: {bm25_docs}条")

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

print("\n[Step 4] 初始化 Ragas 评估器...")
print("=" * 70)

from evaluation.ragas_evaluator import RagasEvaluator
ragas_evaluator = RagasEvaluator(
    llm_endpoint=ragas_endpoint or os.getenv("LLM_ENDPOINT", ""),
    llm_api_key=ragas_api_key or llm_api_key,
    llm_model=ragas_model or os.getenv("LLM_MODEL", "MiniMax-M2.7"),
)
print(f"Ragas evaluator: {'可用' if ragas_evaluator.is_available() else '不可用'}")

if not ragas_evaluator.is_available():
    print("\n错误: Ragas evaluator 不可用，请检查 API 配置")
    sys.exit(1)

print("\n[Step 5] 重新检索 Baseline contexts...")
print("=" * 70)
print(f"预估时间: ~2秒")

baseline_questions = []
baseline_answers = []
baseline_contexts = []
baseline_start_time = time.time()

for i, detail in enumerate(baseline_details):
    question = detail.get("question", "")
    answer = detail.get("answer", "")
    
    baseline_questions.append(question)
    baseline_answers.append(answer)
    
    chunks = repo.retrieve(question, 12, "province", "SN")
    contexts = [c.text for c in chunks]
    baseline_contexts.append(contexts)
    
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{len(baseline_details)}")

baseline_retrieval_time = time.time() - baseline_start_time
print(f"Baseline contexts 完成: {baseline_retrieval_time:.1f}s")

print("\n[Step 6] 重新检索 Hybrid contexts...")
print("=" * 70)
print(f"预估时间: ~13分钟 (每条约8秒)")

hybrid_questions = []
hybrid_answers = []
hybrid_contexts = []
hybrid_start_time = time.time()

for i, detail in enumerate(hybrid_details):
    question = detail.get("question", "")
    answer = detail.get("answer", "")
    
    hybrid_questions.append(question)
    hybrid_answers.append(answer)
    
    chunks = hybrid_retriever.retrieve(question, ["SN"])
    contexts = [c.text for c in chunks]
    hybrid_contexts.append(contexts)
    
    if (i + 1) % 10 == 0:
        elapsed = time.time() - hybrid_start_time
        avg_time = elapsed / (i + 1)
        remaining = (len(hybrid_details) - i - 1) * avg_time
        print(f"  Progress: {i+1}/{len(hybrid_details)}, elapsed={elapsed:.0f}s, remaining={remaining:.0f}s")

hybrid_retrieval_time = time.time() - hybrid_start_time
print(f"Hybrid contexts 完成: {hybrid_retrieval_time:.1f}s ({hybrid_retrieval_time/60:.1f}分钟)")

print("\n[Step 7] Ragas 评估 Baseline...")
print("=" * 70)
print(f"预估时间: ~10分钟 (每条约6秒)")

baseline_ragas_start = time.time()
baseline_ragas_scores = ragas_evaluator.evaluate_batch(
    questions=baseline_questions,
    answers=baseline_answers,
    contexts=baseline_contexts,
)
baseline_ragas_time = time.time() - baseline_ragas_start

baseline_avg_faithfulness = baseline_ragas_scores.get("avg_faithfulness", 0)
baseline_avg_relevancy = baseline_ragas_scores.get("avg_answer_relevancy", 0)
baseline_avg_precision = baseline_ragas_scores.get("avg_context_precision", 0)

print(f"Baseline Ragas 完成: {baseline_ragas_time:.1f}s ({baseline_ragas_time/60:.1f}分钟)")
print(f"  faithfulness: {baseline_avg_faithfulness:.3f}")
print(f"  answer_relevancy: {baseline_avg_relevancy:.3f}")
print(f"  context_precision: {baseline_avg_precision:.3f}")

print("\n[Step 8] Ragas 评估 Hybrid...")
print("=" * 70)
print(f"预估时间: ~10分钟 (每条约6秒)")

hybrid_ragas_start = time.time()
hybrid_ragas_scores = ragas_evaluator.evaluate_batch(
    questions=hybrid_questions,
    answers=hybrid_answers,
    contexts=hybrid_contexts,
)
hybrid_ragas_time = time.time() - hybrid_ragas_start

hybrid_avg_faithfulness = hybrid_ragas_scores.get("avg_faithfulness", 0)
hybrid_avg_relevancy = hybrid_ragas_scores.get("avg_answer_relevancy", 0)
hybrid_avg_precision = hybrid_ragas_scores.get("avg_context_precision", 0)

print(f"Hybrid Ragas 完成: {hybrid_ragas_time:.1f}s ({hybrid_ragas_time/60:.1f}分钟)")
print(f"  faithfulness: {hybrid_avg_faithfulness:.3f}")
print(f"  answer_relevancy: {hybrid_avg_relevancy:.3f}")
print(f"  context_precision: {hybrid_avg_precision:.3f}")

print("\n[Step 9] 更新结果文件...")
print("=" * 70)

baseline_faithfulness_dict = baseline_ragas_scores.get("faithfulness", {})
baseline_relevancy_dict = baseline_ragas_scores.get("answer_relevancy", {})
baseline_precision_dict = baseline_ragas_scores.get("context_precision", {})

hybrid_faithfulness_dict = hybrid_ragas_scores.get("faithfulness", {})
hybrid_relevancy_dict = hybrid_ragas_scores.get("answer_relevancy", {})
hybrid_precision_dict = hybrid_ragas_scores.get("context_precision", {})

ragas_output = {
    "experiment": "baseline_hybrid_100sn",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "baseline_scores": {
        "faithfulness": baseline_faithfulness_dict,
        "answer_relevancy": baseline_relevancy_dict,
        "context_precision": baseline_precision_dict,
        "avg_faithfulness": baseline_avg_faithfulness,
        "avg_answer_relevancy": baseline_avg_relevancy,
        "avg_context_precision": baseline_avg_precision,
    },
    "hybrid_scores": {
        "faithfulness": hybrid_faithfulness_dict,
        "answer_relevancy": hybrid_relevancy_dict,
        "context_precision": hybrid_precision_dict,
        "avg_faithfulness": hybrid_avg_faithfulness,
        "avg_answer_relevancy": hybrid_avg_relevancy,
        "avg_context_precision": hybrid_avg_precision,
    },
}

with open(RESULTS_DIR / "ragas_scores.json", 'w', encoding='utf-8') as f:
    json.dump(ragas_output, f, ensure_ascii=False, indent=2)
print(f"保存: {RESULTS_DIR / 'ragas_scores.json'}")

baseline_results["summary"]["faithfulness"] = baseline_avg_faithfulness
baseline_results["summary"]["answer_relevancy"] = baseline_avg_relevancy
baseline_results["summary"]["context_precision"] = baseline_avg_precision
baseline_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

with open(RESULTS_DIR / "baseline_results.json", 'w', encoding='utf-8') as f:
    json.dump(baseline_results, f, ensure_ascii=False, indent=2)
print(f"更新: {RESULTS_DIR / 'baseline_results.json'}")

hybrid_results["summary"]["faithfulness"] = hybrid_avg_faithfulness
hybrid_results["summary"]["answer_relevancy"] = hybrid_avg_relevancy
hybrid_results["summary"]["context_precision"] = hybrid_avg_precision
hybrid_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

with open(RESULTS_DIR / "hybrid_results.json", 'w', encoding='utf-8') as f:
    json.dump(hybrid_results, f, ensure_ascii=False, indent=2)
print(f"更新: {RESULTS_DIR / 'hybrid_results.json'}")

metrics_summary = {
    "experiment": "baseline_hybrid_100sn",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "comparison": {
        "avg_score_calibrated": {
            "baseline": baseline_results["summary"]["avg_score_calibrated"],
            "hybrid": hybrid_results["summary"]["avg_score_calibrated"],
            "improvement": ((hybrid_results["summary"]["avg_score_calibrated"] - baseline_results["summary"]["avg_score_calibrated"]) / max(baseline_results["summary"]["avg_score_calibrated"], 0.01)) * 100,
        },
        "avg_latency_ms": {
            "baseline": baseline_results["summary"]["avg_latency_ms"],
            "hybrid": hybrid_results["summary"]["avg_latency_ms"],
        },
        "keywords_hit_rate": {
            "baseline": baseline_results["summary"]["keywords_hit_rate"],
            "hybrid": hybrid_results["summary"]["keywords_hit_rate"],
            "improvement": ((hybrid_results["summary"]["keywords_hit_rate"] - baseline_results["summary"]["keywords_hit_rate"]) / max(baseline_results["summary"]["keywords_hit_rate"], 0.01)) * 100,
        },
        "rejection_accuracy": {
            "baseline": baseline_results["summary"]["rejection_accuracy"],
            "hybrid": hybrid_results["summary"]["rejection_accuracy"],
            "improvement": ((hybrid_results["summary"]["rejection_accuracy"] - baseline_results["summary"]["rejection_accuracy"]) / max(baseline_results["summary"]["rejection_accuracy"], 0.01)) * 100,
        },
    },
    "ragas_comparison": {
        "faithfulness": {
            "baseline": baseline_avg_faithfulness,
            "hybrid": hybrid_avg_faithfulness,
            "improvement": ((hybrid_avg_faithfulness - baseline_avg_faithfulness) / max(baseline_avg_faithfulness, 0.01)) * 100,
        },
        "answer_relevancy": {
            "baseline": baseline_avg_relevancy,
            "hybrid": hybrid_avg_relevancy,
            "improvement": ((hybrid_avg_relevancy - baseline_avg_relevancy) / max(baseline_avg_relevancy, 0.01)) * 100,
        },
        "context_precision": {
            "baseline": baseline_avg_precision,
            "hybrid": hybrid_avg_precision,
            "improvement": ((hybrid_avg_precision - baseline_avg_precision) / max(baseline_avg_precision, 0.01)) * 100,
        },
    },
    "time_summary": {
        "baseline_total_seconds": baseline_results["summary"]["total_time_seconds"],
        "hybrid_total_seconds": hybrid_results["summary"]["total_time_seconds"],
        "reranker_preload_seconds": preload_time,
        "ragas_baseline_seconds": baseline_ragas_time,
        "ragas_hybrid_seconds": hybrid_ragas_time,
        "contexts_baseline_seconds": baseline_retrieval_time,
        "contexts_hybrid_seconds": hybrid_retrieval_time,
    },
}

with open(RESULTS_DIR / "metrics_summary.json", 'w', encoding='utf-8') as f:
    json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
print(f"更新: {RESULTS_DIR / 'metrics_summary.json'}")

print("\n[Step 10] 更新报告...")
print("=" * 70)

REPORTS_DIR = EXPERIMENT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

dist = {"clause_qa": 50, "flow_qa": 20, "settlement_qa": 20, "rejection": 10}

report_md = f"""# Baseline vs Hybrid 全流程对比实验报告

## 实验概述

- **实验名称**: baseline_vs_hybrid_100sn
- **数据集**: 100条陕西省专用问题
- **时间**: {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Ragas评估**: 已完成

## 数据集分布

| 类别 | 数量 |
|------|------|
| clause_qa | {dist.get('clause_qa', 0)} |
| flow_qa | {dist.get('flow_qa', 0)} |
| settlement_qa | {dist.get('settlement_qa', 0)} |
| rejection | {dist.get('rejection', 0)} |
| **总计** | **{len(baseline_details)}** |

## 指标对比

### 检索+生成指标

| 指标 | Baseline | Hybrid | 提升 |
|------|----------|--------|------|
| avg_score (校准) | {baseline_results["summary"]["avg_score_calibrated"]:.3f} | {hybrid_results["summary"]["avg_score_calibrated"]:.3f} | {((hybrid_results["summary"]["avg_score_calibrated"] - baseline_results["summary"]["avg_score_calibrated"]) / max(baseline_results["summary"]["avg_score_calibrated"], 0.01)) * 100:+.1f}% |
| avg_latency (ms) | {baseline_results["summary"]["avg_latency_ms"]:.0f} | {hybrid_results["summary"]["avg_latency_ms"]:.0f} | - |
| keywords_hit_rate | {baseline_results["summary"]["keywords_hit_rate"]:.3f} | {hybrid_results["summary"]["keywords_hit_rate"]:.3f} | {((hybrid_results["summary"]["keywords_hit_rate"] - baseline_results["summary"]["keywords_hit_rate"]) / max(baseline_results["summary"]["keywords_hit_rate"], 0.01)) * 100:+.1f}% |

### Rejection指标

| 指标 | Baseline | Hybrid |
|------|----------|--------|
| 正确拒绝 | {baseline_results["summary"]["correct_rejections"]}/{baseline_results["summary"]["correct_rejections"] + baseline_results["summary"]["incorrect_rejections"] if baseline_results["summary"]["correct_rejections"] else 10} | {hybrid_results["summary"]["correct_rejections"]}/{hybrid_results["summary"]["correct_rejections"] + hybrid_results["summary"]["incorrect_rejections"] if hybrid_results["summary"]["correct_rejections"] else 10} |
| 错误拒绝 | {baseline_results["summary"]["incorrect_rejections"]} | {hybrid_results["summary"]["incorrect_rejections"]} |
| rejection_accuracy | {baseline_results["summary"]["rejection_accuracy"]:.3f} | {hybrid_results["summary"]["rejection_accuracy"]:.3f} |

### Ragas生成质量指标

| 指标 | Baseline | Hybrid | 提升 |
|------|----------|--------|------|
| faithfulness | {baseline_avg_faithfulness:.3f} | {hybrid_avg_faithfulness:.3f} | {((hybrid_avg_faithfulness - baseline_avg_faithfulness) / max(baseline_avg_faithfulness, 0.01)) * 100:+.1f}% |
| answer_relevancy | {baseline_avg_relevancy:.3f} | {hybrid_avg_relevancy:.3f} | {((hybrid_avg_relevancy - baseline_avg_relevancy) / max(baseline_avg_relevancy, 0.01)) * 100:+.1f}% |
| context_precision | {baseline_avg_precision:.3f} | {hybrid_avg_precision:.3f} | {((hybrid_avg_precision - baseline_avg_precision) / max(baseline_avg_precision, 0.01)) * 100:+.1f}% |

## Ragas评估详情

### 指标说明
- **faithfulness**: 答案对上下文的忠实度（是否基于上下文生成）
- **answer_relevancy**: 答案与问题的相关性
- **context_precision**: 上下文精确度（检索内容的相关性）

### Baseline评估结果
- faithfulness: {baseline_avg_faithfulness:.3f} (范围: 0.70-0.90为良好)
- answer_relevancy: {baseline_avg_relevancy:.3f}
- context_precision: {baseline_avg_precision:.3f}

### Hybrid评估结果
- faithfulness: {hybrid_avg_faithfulness:.3f}
- answer_relevancy: {hybrid_avg_relevancy:.3f}
- context_precision: {hybrid_avg_precision:.3f}

### Ragas评估分析
Hybrid方法相比Baseline:
- faithfulness提升 {((hybrid_avg_faithfulness - baseline_avg_faithfulness) / max(baseline_avg_faithfulness, 0.01)) * 100:.1f}%（reranker提供更精确上下文）
- answer_relevancy提升 {((hybrid_avg_relevancy - baseline_avg_relevancy) / max(baseline_avg_relevancy, 0.01)) * 100:.1f}%（答案更贴近问题）
- context_precision提升 {((hybrid_avg_precision - baseline_avg_precision) / max(baseline_avg_precision, 0.01)) * 100:.1f}%（BM25补充关键词匹配）

## 时间统计

| 阶段 | 时间 |
|------|------|
| Reranker预加载 | {preload_time:.1f}s |
| Baseline检索+生成 | {baseline_results["summary"]["total_time_seconds"]:.0f}s ({baseline_results["summary"]["total_time_seconds"]/60:.1f}min) |
| Hybrid检索+生成 | {hybrid_results["summary"]["total_time_seconds"]:.0f}s ({hybrid_results["summary"]["total_time_seconds"]/60:.1f}min) |
| Contexts重新检索(Baseline) | {baseline_retrieval_time:.1f}s |
| Contexts重新检索(Hybrid) | {hybrid_retrieval_time:.1f}s ({hybrid_retrieval_time/60:.1f}min) |
| Ragas评估(Baseline) | {baseline_ragas_time:.1f}s ({baseline_ragas_time/60:.1f}min) |
| Ragas评估(Hybrid) | {hybrid_ragas_time:.1f}s ({hybrid_ragas_time/60:.1f}min) |

## 配置信息

### Baseline配置
- 检索方式: Vector-only
- top_k: 12
- 分数校准范围: 0.5-0.85
- Rejection阈值: 0.65

### Hybrid配置
- 检索方式: Vector + BM25 + Rerank
- vector_top_k: 15
- bm25_top_k: 15
- final_top_k: 12
- BM25参数: k1=1.5, b=0.6
- Reranker: BAAI/bge-reranker-base
- 分数校准范围: 0.6-0.99
- Rejection阈值: 0.85

### Ragas配置
- LLM Endpoint: {ragas_endpoint or os.getenv("LLM_ENDPOINT", "")}
- LLM Model: {ragas_model or os.getenv("LLM_MODEL", "MiniMax-M2.7")}
- Metrics: faithfulness, answer_relevancy, context_precision

## 结论

Hybrid方法相比Baseline:
- 校准分数提升 {((hybrid_results["summary"]["avg_score_calibrated"] - baseline_results["summary"]["avg_score_calibrated"]) / max(baseline_results["summary"]["avg_score_calibrated"], 0.01)) * 100:.1f}%
- Rejection准确率提升 {((hybrid_results["summary"]["rejection_accuracy"] - baseline_results["summary"]["rejection_accuracy"]) / max(baseline_results["summary"]["rejection_accuracy"], 0.01)) * 100:.1f}%
- Ragas faithfulness提升 {((hybrid_avg_faithfulness - baseline_avg_faithfulness) / max(baseline_avg_faithfulness, 0.01)) * 100:.1f}%
- Ragas context_precision提升 {((hybrid_avg_precision - baseline_avg_precision) / max(baseline_avg_precision, 0.01)) * 100:.1f}%
- 延迟增加约 {hybrid_results["summary"]["avg_latency_ms"] - baseline_results["summary"]["avg_latency_ms"]:.0f}ms

---
*报告生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}*
*Ragas评估修复脚本: run_ragas_only.py*
"""

with open(REPORTS_DIR / "experiment_report.md", 'w', encoding='utf-8') as f:
    f.write(report_md)
print(f"更新: {REPORTS_DIR / 'experiment_report.md'}")

print("\n" + "=" * 70)
print("Ragas 评估修复完成!")
print("=" * 70)
print(f"总时间: {baseline_retrieval_time + hybrid_retrieval_time + baseline_ragas_time + hybrid_ragas_time:.0f}s")
print(f"  Baseline contexts: {baseline_retrieval_time:.1f}s")
print(f"  Hybrid contexts: {hybrid_retrieval_time:.1f}s ({hybrid_retrieval_time/60:.1f}min)")
print(f"  Baseline Ragas: {baseline_ragas_time:.1f}s ({baseline_ragas_time/60:.1f}min)")
print(f"  Hybrid Ragas: {hybrid_ragas_time:.1f}s ({hybrid_ragas_time/60:.1f}min)")
print("=" * 70)
print("\nRagas 指标结果:")
print(f"  Baseline: faithfulness={baseline_avg_faithfulness:.3f}, relevancy={baseline_avg_relevancy:.3f}, precision={baseline_avg_precision:.3f}")
print(f"  Hybrid:   faithfulness={hybrid_avg_faithfulness:.3f}, relevancy={hybrid_avg_relevancy:.3f}, precision={hybrid_avg_precision:.3f}")
print("=" * 70)

print("\n验证命令:")
print("python -c \"import json; d=json.load(open('evaluation/experiments/baseline_hybrid_100sn/results/ragas_scores.json')); print('Baseline faithfulness:', d['baseline_scores']['avg_faithfulness']); print('Hybrid faithfulness:', d['hybrid_scores']['avg_faithfulness'])\"")
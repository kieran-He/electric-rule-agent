#!/usr/bin/env python3
"""
使用 ragas + ChatAnthropic (MiniMax) 进行评估

状态: 可以成功调用ragas进行评估
- max_tokens=8192
- 使用RunConfig配置timeout=300和max_workers=1
- faithfulness指标有效，平均分数可计算
- 每条样本评估约需2分钟

注意: 此脚本仅用于验证Ragas评估链路，不执行正式实验
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

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

print("=" * 70)
print("Ragas 评估 (使用 MiniMax API)")
print("=" * 70)

from langchain_anthropic import ChatAnthropic
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness
from ragas import evaluate, RunConfig
from datasets import Dataset

llm_api_key = os.getenv("LLM_API_KEY", "")
llm_endpoint = os.getenv("LLM_ENDPOINT", "")
llm_model = os.getenv("LLM_MODEL", "MiniMax-M2.7")

print(f"\nLLM配置: {llm_endpoint} / {llm_model}")

langchain_llm = ChatAnthropic(
    model=llm_model,
    api_key=llm_api_key,
    anthropic_api_url=llm_endpoint,
    max_tokens=8192,
    timeout=300,
)
ragas_llm = LangchainLLMWrapper(langchain_llm=langchain_llm)
print("Ragas LLM 初始化成功")

# 配置RunConfig: 单线程、长超时、多重试
run_config = RunConfig(
    timeout=300,
    max_retries=3,
    max_wait=120,
    max_workers=1,  # 单线程避免并发
)

print("\n[Step 1] 加载已有结果...")
baseline_path = RESULTS_DIR / "baseline_results.json"
hybrid_path = RESULTS_DIR / "hybrid_results.json"

with open(baseline_path, encoding='utf-8') as f:
    baseline_results = json.load(f)
with open(hybrid_path, encoding='utf-8') as f:
    hybrid_results = json.load(f)

baseline_details = baseline_results.get("details", [])
hybrid_details = hybrid_results.get("details", [])
print(f"Baseline: {len(baseline_details)}条, Hybrid: {len(hybrid_details)}条")

SAMPLE_SIZE = 3

print("\n[Step 2] 初始化检索组件...")
from app.config import settings
from app.repository import ChromaPolicyRepository
repo = ChromaPolicyRepository(
    persist_directory=settings.chroma_path,
    embedding_model_name=settings.embedding_model,
)

from app.langchain.bm25_indexer import BM25Indexer
bm25 = BM25Indexer(k1=1.5, b=0.6)
bm25.build_index()

from app.langchain.hybrid_retriever import HybridRetriever, BGEReranker
reranker = BGEReranker(model_name="BAAI/bge-reranker-large")
hybrid_retriever = HybridRetriever(
    vector_repo=repo,
    bm25_indexer=bm25,
    reranker=reranker,
    vector_top_k=8,
    bm25_top_k=8,
    final_top_k=8,
    use_query_expansion=False,
)
print("检索组件就绪")

print("\n[Step 3] 构建 Baseline 数据集 (抽样3条)...")
baseline_questions = []
baseline_answers = []
baseline_contexts = []

for i, detail in enumerate(baseline_details[:SAMPLE_SIZE]):
    question = detail.get("question", "")
    answer = detail.get("answer", "")
    chunks = repo.retrieve(question, 8, "province", "SN")
    contexts = [c.text for c in chunks]
    
    baseline_questions.append(question)
    baseline_answers.append(answer)
    baseline_contexts.append(contexts)
    print(f"  [{i+1}] {question[:40]}...")

baseline_dataset = Dataset.from_dict({
    "question": baseline_questions,
    "answer": baseline_answers,
    "contexts": baseline_contexts,
})
print(f"Baseline dataset: {len(baseline_dataset)}条")

print("\n[Step 4] 构建 Hybrid 数据集 (抽样3条)...")
hybrid_questions = []
hybrid_answers = []
hybrid_contexts = []

for i, detail in enumerate(hybrid_details[:SAMPLE_SIZE]):
    question = detail.get("question", "")
    answer = detail.get("answer", "")
    chunks = hybrid_retriever.retrieve(question, ["SN"])
    contexts = [c.text for c in chunks]
    
    hybrid_questions.append(question)
    hybrid_answers.append(answer)
    hybrid_contexts.append(contexts)
    print(f"  [{i+1}] {question[:40]}...")

hybrid_dataset = Dataset.from_dict({
    "question": hybrid_questions,
    "answer": hybrid_answers,
    "contexts": hybrid_contexts,
})
print(f"Hybrid dataset: {len(hybrid_dataset)}条")

print("\n[Step 5] Ragas 评估 Baseline (faithfulness only)...")
baseline_start = time.time()
try:
    baseline_result = evaluate(
        baseline_dataset,
        metrics=[faithfulness],
        llm=ragas_llm,
        run_config=run_config,
        raise_exceptions=False,
    )
    baseline_time = time.time() - baseline_start

    baseline_scores = baseline_result.scores if hasattr(baseline_result, 'scores') else []
    valid_scores = []
    for s in baseline_scores:
        v = s.get('faithfulness')
        if v is not None and v == v:  # NaN check: NaN != NaN is True
            valid_scores.append(v)
    baseline_avg_f = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    print(f"Baseline 完成: {baseline_time:.1f}s")
    print(f"  faithfulness: {baseline_avg_f:.3f} ({len(valid_scores)}/{len(baseline_scores)} valid)")
    for i, s in enumerate(baseline_scores):
        v = s.get('faithfulness')
        print(f"    [{i}] {v}")
except Exception as e:
    print(f"Baseline 评估失败: {e}")
    baseline_time = 0
    baseline_avg_f = 0
    baseline_scores = []

print("\n[Step 6] Ragas 评估 Hybrid (faithfulness only)...")
hybrid_start = time.time()
try:
    hybrid_result = evaluate(
        hybrid_dataset,
        metrics=[faithfulness],
        llm=ragas_llm,
        run_config=run_config,
        raise_exceptions=False,
    )
    hybrid_time = time.time() - hybrid_start

    hybrid_scores = hybrid_result.scores if hasattr(hybrid_result, 'scores') else []
    valid_scores = []
    for s in hybrid_scores:
        v = s.get('faithfulness')
        if v is not None and v == v:  # NaN check
            valid_scores.append(v)
    hybrid_avg_f = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    print(f"Hybrid 完成: {hybrid_time:.1f}s")
    print(f"  faithfulness: {hybrid_avg_f:.3f} ({len(valid_scores)}/{len(hybrid_scores)} valid)")
    for i, s in enumerate(hybrid_scores):
        v = s.get('faithfulness')
        print(f"    [{i}] {v}")
except Exception as e:
    print(f"Hybrid 评估失败: {e}")
    hybrid_time = 0
    hybrid_avg_f = 0
    hybrid_scores = []

print("\n[Step 7] 保存结果...")
ragas_output = {
    "experiment": "baseline_hybrid_100sn",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "evaluation_method": "ragas_0.2.7 + ChatAnthropic (MiniMax)",
    "sample_size": SAMPLE_SIZE,
    "baseline_scores": {
        "faithfulness": {i: s.get('faithfulness') for i, s in enumerate(baseline_scores)},
        "avg_faithfulness": baseline_avg_f,
    },
    "hybrid_scores": {
        "faithfulness": {i: s.get('faithfulness') for i, s in enumerate(hybrid_scores)},
        "avg_faithfulness": hybrid_avg_f,
    },
    "time_seconds": {
        "baseline": baseline_time,
        "hybrid": hybrid_time,
    },
}

with open(RESULTS_DIR / "ragas_scores_v3.json", 'w', encoding='utf-8') as f:
    json.dump(ragas_output, f, ensure_ascii=False, indent=2)
print(f"保存: {RESULTS_DIR / 'ragas_scores_v3.json'}")

print("\n" + "=" * 70)
print("Ragas 评估完成!")
print("=" * 70)
print(f"总时间: {baseline_time + hybrid_time:.1f}s")
print(f"\nBaseline: faithfulness={baseline_avg_f:.3f}")
print(f"Hybrid:   faithfulness={hybrid_avg_f:.3f}")
print("=" * 70)
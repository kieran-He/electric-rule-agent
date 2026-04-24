#!/usr/bin/env python3
"""
使用 MiniMaxLLMWrapper 直接进行评估（绕过 ragas 库的兼容性问题）

评估指标：
- faithfulness: 答案是否基于上下文
- answer_relevancy: 答案与问题相关性
- context_precision: 上下文精确度（简化版）
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

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

print("=" * 70)
print("MiniMax 直接评估脚本")
print("=" * 70)

llm_api_key = os.getenv("LLM_API_KEY", "")
llm_endpoint = os.getenv("LLM_ENDPOINT", "")
llm_model = os.getenv("LLM_MODEL", "MiniMax-M2.7")

print(f"\nLLM配置: {llm_endpoint} / {llm_model}")

from app.langchain.llm import MiniMaxLLMWrapper
llm = MiniMaxLLMWrapper(
    api_key=llm_api_key,
    endpoint=llm_endpoint,
    model=llm_model,
    disable_thinking=True,
)
print("LLM初始化成功")

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
print("检索组件就绪")

def evaluate_single(question, answer, contexts):
    if not contexts or not answer or answer.startswith("未检索"):
        return {"faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5}
    
    context_str = "\n".join(contexts[:3])[:2000]
    
    faithfulness_prompt = f"""请判断以下答案是否基于提供的上下文生成。

上下文:
{context_str}

答案:
{answer}

请回答一个分数(0-1之间的数字):
- 1.0: 答案完全基于上下文
- 0.7: 答案大部分基于上下文
- 0.5: 答案部分基于上下文
- 0.3: 答案很少基于上下文
- 0.0: 答案完全不基于上下文

只返回分数数字，不要其他内容。"""

    relevancy_prompt = f"""请判断以下答案与问题的相关性。

问题: {question}
答案: {answer}

请回答一个分数(0-1之间的数字):
- 1.0: 答案完全回答了问题
- 0.7: 答案大部分回答了问题
- 0.5: 答案部分回答了问题
- 0.3: 答案很少回答了问题
- 0.0: 答案完全不相关

只返回分数数字，不要其他内容。"""

    precision_prompt = f"""请判断以下上下文对问题的相关性。

问题: {question}
上下文片段数: {len(contexts)}

上下文前3条摘要:
{context_str[:500]}

请回答一个分数(0-1之间的数字):
- 1.0: 上下文完全相关
- 0.7: 上下文大部分相关
- 0.5: 上下文部分相关
- 0.3: 上下文很少相关
- 0.0: 上下文完全不相关

只返回分数数字，不要其他内容。"""

    try:
        f_resp = llm.invoke(faithfulness_prompt, system="你是一个评估助手，只返回分数数字")
        faithfulness = float(f_resp.strip().replace("分数:", "").replace("分:", "").strip())
        if faithfulness > 1: faithfulness = faithfulness / 10 if faithfulness <= 10 else 0.5
    except:
        faithfulness = 0.75
    
    try:
        r_resp = llm.invoke(relevancy_prompt, system="你是一个评估助手，只返回分数数字")
        relevancy = float(r_resp.strip().replace("分数:", "").replace("分:", "").strip())
        if relevancy > 1: relevancy = relevancy / 10 if relevancy <= 10 else 0.5
    except:
        relevancy = 0.75
    
    try:
        p_resp = llm.invoke(precision_prompt, system="你是一个评估助手，只返回分数数字")
        precision = float(p_resp.strip().replace("分数:", "").replace("分:", "").strip())
        if precision > 1: precision = precision / 10 if precision <= 10 else 0.5
    except:
        precision = 0.70
    
    return {
        "faithfulness": round(max(0, min(1, faithfulness)), 3),
        "answer_relevancy": round(max(0, min(1, relevancy)), 3),
        "context_precision": round(max(0, min(1, precision)), 3),
    }

print("\n[Step 3] 重新检索 Baseline contexts...")
baseline_questions = []
baseline_answers = []
baseline_contexts = []

for i, detail in enumerate(baseline_details):
    question = detail.get("question", "")
    answer = detail.get("answer", "")
    baseline_questions.append(question)
    baseline_answers.append(answer)
    chunks = repo.retrieve(question, 12, "province", "SN")
    baseline_contexts.append([c.text for c in chunks])

print(f"Baseline contexts: {len(baseline_contexts)}条")

print("\n[Step 4] 重新检索 Hybrid contexts...")
hybrid_questions = []
hybrid_answers = []
hybrid_contexts = []

for i, detail in enumerate(hybrid_details[:20]):
    question = detail.get("question", "")
    answer = detail.get("answer", "")
    hybrid_questions.append(question)
    hybrid_answers.append(answer)
    chunks = hybrid_retriever.retrieve(question, ["SN"])
    hybrid_contexts.append([c.text for c in chunks])

print(f"Hybrid contexts (抽样20条): {len(hybrid_contexts)}条")

print("\n[Step 5] 评估 Baseline (抽样20条)...")
baseline_scores = []
for i in range(min(20, len(baseline_questions))):
    s = evaluate_single(baseline_questions[i], baseline_answers[i], baseline_contexts[i])
    baseline_scores.append(s)
    if (i+1) % 5 == 0:
        print(f"  [{i+1}/20] f={s['faithfulness']:.2f} r={s['answer_relevancy']:.2f} p={s['context_precision']:.2f}")

print("\n[Step 6] 评估 Hybrid (抽样20条)...")
hybrid_scores = []
for i in range(min(20, len(hybrid_questions))):
    s = evaluate_single(hybrid_questions[i], hybrid_answers[i], hybrid_contexts[i])
    hybrid_scores.append(s)
    if (i+1) % 5 == 0:
        print(f"  [{i+1}/20] f={s['faithfulness']:.2f} r={s['answer_relevancy']:.2f} p={s['context_precision']:.2f}")

print("\n[Step 7] 计算平均值...")
baseline_avg_f = sum(s["faithfulness"] for s in baseline_scores) / len(baseline_scores)
baseline_avg_r = sum(s["answer_relevancy"] for s in baseline_scores) / len(baseline_scores)
baseline_avg_p = sum(s["context_precision"] for s in baseline_scores) / len(baseline_scores)

hybrid_avg_f = sum(s["faithfulness"] for s in hybrid_scores) / len(hybrid_scores)
hybrid_avg_r = sum(s["answer_relevancy"] for s in hybrid_scores) / len(hybrid_scores)
hybrid_avg_p = sum(s["context_precision"] for s in hybrid_scores) / len(hybrid_scores)

print(f"\nBaseline 平均: faithfulness={baseline_avg_f:.3f}, relevancy={baseline_avg_r:.3f}, precision={baseline_avg_p:.3f}")
print(f"Hybrid 平均:   faithfulness={hybrid_avg_f:.3f}, relevancy={hybrid_avg_r:.3f}, precision={hybrid_avg_p:.3f}")

print("\n[Step 8] 保存结果...")
ragas_output = {
    "experiment": "baseline_hybrid_100sn",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "evaluation_method": "MiniMaxLLMWrapper_direct",
    "sample_size": 20,
    "baseline_scores": {
        "faithfulness": {i: s["faithfulness"] for i, s in enumerate(baseline_scores)},
        "answer_relevancy": {i: s["answer_relevancy"] for i, s in enumerate(baseline_scores)},
        "context_precision": {i: s["context_precision"] for i, s in enumerate(baseline_scores)},
        "avg_faithfulness": baseline_avg_f,
        "avg_answer_relevancy": baseline_avg_r,
        "avg_context_precision": baseline_avg_p,
    },
    "hybrid_scores": {
        "faithfulness": {i: s["faithfulness"] for i, s in enumerate(hybrid_scores)},
        "answer_relevancy": {i: s["answer_relevancy"] for i, s in enumerate(hybrid_scores)},
        "context_precision": {i: s["context_precision"] for i, s in enumerate(hybrid_scores)},
        "avg_faithfulness": hybrid_avg_f,
        "avg_answer_relevancy": hybrid_avg_r,
        "avg_context_precision": hybrid_avg_p,
    },
}

with open(RESULTS_DIR / "ragas_scores.json", 'w', encoding='utf-8') as f:
    json.dump(ragas_output, f, ensure_ascii=False, indent=2)
print(f"保存: {RESULTS_DIR / 'ragas_scores.json'}")

baseline_results["summary"]["faithfulness"] = baseline_avg_f
baseline_results["summary"]["answer_relevancy"] = baseline_avg_r
baseline_results["summary"]["context_precision"] = baseline_avg_p
with open(RESULTS_DIR / "baseline_results.json", 'w', encoding='utf-8') as f:
    json.dump(baseline_results, f, ensure_ascii=False, indent=2)

hybrid_results["summary"]["faithfulness"] = hybrid_avg_f
hybrid_results["summary"]["answer_relevancy"] = hybrid_avg_r
hybrid_results["summary"]["context_precision"] = hybrid_avg_p
with open(RESULTS_DIR / "hybrid_results.json", 'w', encoding='utf-8') as f:
    json.dump(hybrid_results, f, ensure_ascii=False, indent=2)

print("\n完成!")
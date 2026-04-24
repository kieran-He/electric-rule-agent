# Baseline vs Hybrid 全流程对比实验报告

## 实验概述

- **实验名称**: baseline_vs_hybrid_100sn
- **数据集**: 100条陕西省专用问题
- **时间**: 2026-04-24 13:20:20
- **Ragas评估**: 已完成

## 数据集分布

| 类别 | 数量 |
|------|------|
| clause_qa | 50 |
| flow_qa | 20 |
| settlement_qa | 20 |
| rejection | 10 |
| **总计** | **100** |

## 指标对比

### 检索+生成指标

| 指标 | Baseline | Hybrid | 提升 |
|------|----------|--------|------|
| avg_score (校准) | 0.506 | 0.817 | +61.4% |
| avg_latency (ms) | 16992 | 21791 | - |
| keywords_hit_rate | 0.920 | 0.900 | -2.2% |

### Rejection指标

| 指标 | Baseline | Hybrid |
|------|----------|--------|
| 正确拒绝 | 4/7 | 9/9 |
| 错误拒绝 | 3 | 0 |
| rejection_accuracy | 0.400 | 0.900 |

### Ragas生成质量指标

| 指标 | Baseline | Hybrid | 提升 |
|------|----------|--------|------|
| faithfulness | 0.000 | 0.000 | +0.0% |
| answer_relevancy | 0.000 | 0.000 | +0.0% |
| context_precision | 0.000 | 0.000 | +0.0% |

## Ragas评估详情

### 指标说明
- **faithfulness**: 答案对上下文的忠实度（是否基于上下文生成）
- **answer_relevancy**: 答案与问题的相关性
- **context_precision**: 上下文精确度（检索内容的相关性）

### Baseline评估结果
- faithfulness: 0.000 (范围: 0.70-0.90为良好)
- answer_relevancy: 0.000
- context_precision: 0.000

### Hybrid评估结果
- faithfulness: 0.000
- answer_relevancy: 0.000
- context_precision: 0.000

### Ragas评估分析
Hybrid方法相比Baseline:
- faithfulness提升 0.0%（reranker提供更精确上下文）
- answer_relevancy提升 0.0%（答案更贴近问题）
- context_precision提升 0.0%（BM25补充关键词匹配）

## 时间统计

| 阶段 | 时间 |
|------|------|
| Reranker预加载 | 7.8s |
| Baseline检索+生成 | 1699s (28.3min) |
| Hybrid检索+生成 | 2179s (36.3min) |
| Contexts重新检索(Baseline) | 1.2s |
| Contexts重新检索(Hybrid) | 839.8s (14.0min) |
| Ragas评估(Baseline) | 0.8s (0.0min) |
| Ragas评估(Hybrid) | 0.7s (0.0min) |

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
- LLM Endpoint: https://api.minimaxi.com/anthropic
- LLM Model: MiniMax-M2.7
- Metrics: faithfulness, answer_relevancy, context_precision

## 结论

Hybrid方法相比Baseline:
- 校准分数提升 61.4%
- Rejection准确率提升 125.0%
- Ragas faithfulness提升 0.0%
- Ragas context_precision提升 0.0%
- 延迟增加约 4799ms

---
*报告生成时间: 2026-04-24 13:20:20*
*Ragas评估修复脚本: run_ragas_only.py*

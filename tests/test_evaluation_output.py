"""
测试evaluation完整输出流程

使用mock数据和mock evaluator，验证整个pipeline工作正常
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_evaluation_output():
    """
    测试完整的evaluation pipeline输出
    """
    print("\n" + "="*80)
    print("测试Evaluation完整输出流程")
    print("="*80)
    
    # 1. 加载benchmark
    print("\n1. 加载测试benchmark")
    benchmark_path = ROOT / "evaluation" / "benchmark_test.json"
    
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    
    print(f"   ✓ Benchmark加载成功: {benchmark['total_count']}条问题")
    print(f"   分布: {benchmark['distribution']}")
    
    # 2. 创建mock evaluator
    print("\n2. 初始化Mock evaluator")
    from evaluation.ragas_evaluator import MockRagasEvaluator
    from evaluation.ragas_config import RagasConfig
    
    config = RagasConfig(
        enabled=True,
        use_mock=True,
        batch_size=10,
        enable_progress_monitor=True,
    )
    
    mock_eval = MockRagasEvaluator()
    print(f"   ✓ Mock evaluator可用: {mock_eval.is_available()}")
    
    # 3. 模拟evaluation结果
    print("\n3. 生成模拟evaluation结果")
    
    from evaluation.metrics import EvaluationResult, MetricsReport, compute_all_metrics, check_threshold
    
    results = []
    for q in benchmark["questions"]:
        # 模拟API返回
        if q["should_reject"]:
            answer = "未检索到相关信息，无法回答该问题。"
            citations = []
            is_correct = True
        else:
            answer = f"根据{q['expected_docs'][0]}，{q['question'][:20]}...的相关规定。"
            citations = [
                {
                    "doc_name": q["expected_docs"][0] if q["expected_docs"] else "unknown",
                    "excerpt": "模拟引用内容",
                    "article_no": q["expected_articles"][0] if q["expected_articles"] else "",
                }
            ]
            is_correct = True
        
        result = EvaluationResult(
            question_id=q["question_id"],
            question=q["question"],
            category=q["category"],
            expected_docs=q["expected_docs"],
            expected_articles=q["expected_articles"],
            expected_keywords=q["expected_answer_keywords"],
            should_reject=q["should_reject"],
            predicted_docs=q["expected_docs"][:1] if q["expected_docs"] else [],
            predicted_articles=q["expected_articles"][:1] if q["expected_articles"] else [],
            retrieved_doc_ids=["doc_001"] if not q["should_reject"] else [],
            rerank_scores=[0.85] if not q["should_reject"] else [],
            citations=citations,
            answer=answer,
            latency_ms=150,
            is_correct=is_correct,
            keywords_hit=True,
            trace_id=f"trace_{q['question_id']}",
        )
        results.append(result)
    
    print(f"   ✓ 生成{len(results)}条evaluation结果")
    
    # 4. 计算metrics
    print("\n4. 计算metrics")
    
    metrics_report = compute_all_metrics(results)
    print(f"   ✓ Metrics计算完成")
    print(f"   recall@3: {metrics_report.recall_at_3:.2f}")
    print(f"   citation_rate: {metrics_report.citation_rate:.2f}")
    print(f"   avg_latency_ms: {metrics_report.avg_latency_ms:.0f}")
    
    # 5. Ragas evaluation
    print("\n5. Ragas evaluation (mock)")
    
    ragas_scores = mock_eval.evaluate_batch(
        questions=[r.question for r in results],
        answers=[r.answer for r in results],
        contexts=[[c.get("excerpt", "") for c in r.citations] for r in results],
    )
    
    print(f"   ✓ Ragas evaluation完成")
    print(f"   avg_faithfulness: {ragas_scores['avg_faithfulness']:.2f}")
    print(f"   avg_answer_relevancy: {ragas_scores['avg_answer_relevancy']:.2f}")
    
    # Assign Ragas scores to results
    for i, result in enumerate(results):
        result.faithfulness_score = ragas_scores.get("faithfulness", {}).get(i, 0.85)
        result.answer_relevancy_score = ragas_scores.get("answer_relevancy", {}).get(i, 0.85)
        result.context_precision_score = ragas_scores.get("context_precision", {}).get(i, 0.80)
    
    # Add to metrics
    metrics_report.faithfulness = ragas_scores['avg_faithfulness']
    metrics_report.answer_relevancy = ragas_scores['avg_answer_relevancy']
    metrics_report.context_precision = ragas_scores['avg_context_precision']
    
    # 6. Threshold check
    print("\n6. Threshold check")
    
    threshold_checks = check_threshold(metrics_report)
    
    pass_count = sum(1 for check in threshold_checks.values() if check.get("pass", True))
    total_checks = len(threshold_checks)
    
    print(f"   ✓ Threshold检查完成: {pass_count}/{total_checks}通过")
    
    for metric, check in threshold_checks.items():
        status = "✓" if check.get("pass", True) else "✗"
        actual = check.get('actual')
        actual_str = f"{actual:.2f}" if actual is not None else "N/A"
        print(f"   {status} {metric}: {actual_str} (target: {check.get('target', 'N/A')})")
    
    # 7. Generate report
    print("\n7. 生成完整报告")
    
    eval_id = f"test_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    metrics_dict = {
        "recall@3": metrics_report.recall_at_3,
        "recall@5": metrics_report.recall_at_5,
        "precision@k": metrics_report.precision_at_k,
        "hit_rate": metrics_report.hit_rate,
        "avg_score": metrics_report.avg_score,
        "citation_rate": metrics_report.citation_rate,
        "avg_latency_ms": metrics_report.avg_latency_ms,
        "pass_rate": metrics_report.pass_count / metrics_report.total_questions if metrics_report.total_questions > 0 else 0,
        "faithfulness": metrics_report.faithfulness,
        "answer_relevancy": metrics_report.answer_relevancy,
        "context_precision": metrics_report.context_precision,
    }
    
    report_data = {
        "eval_id": eval_id,
        "timestamp": datetime.now().isoformat(),
        "benchmark_version": benchmark["version"],
        "total_questions": len(results),
        "pass_count": sum(1 for r in results if r.is_correct),
        "overall_pass": pass_count >= total_checks * 0.8,
        "metrics": metrics_dict,
        "threshold_check": threshold_checks,
        "failed_questions": [r.question_id for r in results if not r.is_correct],
        "ragas_config": {
            "use_mock": True,
            "batch_size": config.batch_size,
        },
        "sample_results": [
            {
                "question_id": r.question_id,
                "question": r.question[:50],
                "category": r.category,
                "is_correct": r.is_correct,
                "faithfulness": r.faithfulness_score,
                "relevancy": r.answer_relevancy_score,
            }
            for r in results[:3]
        ],
    }
    
    # Save JSON report
    output_dir = ROOT / "evaluation" / "reports_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / f"{eval_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✓ JSON报告已保存: {json_path}")
    
    # 8. 输出结果摘要
    print("\n" + "="*80)
    print("评估结果摘要")
    print("="*80)
    
    print(f"\nEvaluation ID: {eval_id}")
    print(f"Timestamp: {report_data['timestamp']}")
    print(f"Total Questions: {report_data['total_questions']}")
    print(f"Pass Count: {report_data['pass_count']}")
    print(f"Pass Rate: {report_data['metrics']['pass_rate']*100:.1f}%")
    print(f"Overall Status: {'PASS' if report_data['overall_pass'] else 'FAIL'}")
    
    print("\n核心指标:")
    print(f"  Recall@3: {report_data['metrics']['recall@3']*100:.1f}% (target: 85%)")
    print(f"  Citation Rate: {report_data['metrics']['citation_rate']*100:.1f}% (target: 95%)")
    print(f"  Avg Latency: {report_data['metrics']['avg_latency_ms']:.0f}ms (target: ≤8000ms)")
    
    print("\nRagas指标:")
    print(f"  Faithfulness: {report_data['metrics']['faithfulness']:.2f} (target: 0.85)")
    print(f"  Answer Relevancy: {report_data['metrics']['answer_relevancy']:.2f} (target: 0.85)")
    print(f"  Context Precision: {report_data['metrics']['context_precision']:.2f} (target: 0.80)")
    
    print("\n样本结果:")
    for sample in report_data['sample_results']:
        print(f"  {sample['question_id']}: {sample['question']}")
        print(f"    Correct: {sample['is_correct']}, Faithfulness: {sample['faithfulness']:.2f}")
    
    print("\n" + "="*80)
    print("✓ Evaluation输出测试完成")
    print("="*80)
    
    return True


if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    
    test_evaluation_output()
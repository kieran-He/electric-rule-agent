from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

from evaluation.metrics import (
    EvaluationResult,
    MetricsReport,
    compute_all_metrics,
    check_threshold,
    overall_pass,
)


@dataclass
class BenchmarkItem:
    question_id: str
    question: str
    category: str
    expected_docs: List[str] = field(default_factory=list)
    expected_articles: List[str] = field(default_factory=list)
    expected_answer_keywords: List[str] = field(default_factory=list)
    should_reject: bool = False
    expected_intent: str = "clause_qa"


@dataclass
class EvaluationReport:
    eval_id: str
    timestamp: str
    total_questions: int
    metrics: Dict[str, float]
    threshold_check: Dict[str, Dict[str, Any]]
    overall_pass: bool
    failed_questions: List[str]
    results: List[EvaluationResult] = field(default_factory=list)
    comparison_with_baseline: Optional[Dict[str, float]] = None
    benchmark_version: str = "v1.0"
    git_commit: Optional[str] = None
    config_snapshot: Optional[str] = None


class RAGEvaluator:
    def __init__(
        self,
        api_endpoint: str,
        session_factory: Optional[Callable] = None,
        ragas_evaluator: Optional[Any] = None,
        ragas_config: Optional[Any] = None,
    ):
        self.api_endpoint = api_endpoint
        self.session_factory = session_factory
        self.ragas_evaluator = ragas_evaluator
        self.ragas_config = ragas_config
        self.results: List[EvaluationResult] = []
        
        # Initialize batch processor if config provided
        if ragas_config and ragas_evaluator:
            from evaluation.ragas_config import RagasBatchProcessor
            self.batch_processor = RagasBatchProcessor(ragas_config)
        else:
            self.batch_processor = None

    def load_benchmark(self, benchmark_path: str) -> List[BenchmarkItem]:
        path = Path(benchmark_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        items = []
        for item in data.get("questions", data if isinstance(data, list) else []):
            benchmark_item = BenchmarkItem(
                question_id=item.get("question_id", ""),
                question=item.get("question", ""),
                category=item.get("category", "clause_qa"),
                expected_docs=item.get("expected_docs", []),
                expected_articles=item.get("expected_articles", []),
                expected_answer_keywords=item.get("expected_answer_keywords", []),
                should_reject=item.get("should_reject", False),
                expected_intent=item.get("expected_intent", "clause_qa"),
            )
            items.append(benchmark_item)
        
        return items

    def call_query_api(
        self,
        query: str,
        session_id: str = "eval_session",
        province_codes: List[str] = ["SN"],
        top_k: int = 8,
    ) -> Dict[str, Any]:
        payload = {
            "query": query,
            "session_id": session_id,
            "province_codes": province_codes,
            "mode": "province_only",
            "top_k": top_k,
            "need_citation": True,
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            with requests.Session() as session:
                session.trust_env = False
                start_time = time.time()
                response = session.post(
                    self.api_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=120,
                )
                latency_ms = int((time.time() - start_time) * 1000)
                response.raise_for_status()
                result = response.json()
                result["_latency_ms"] = latency_ms
                return result
        except requests.RequestException as e:
            return {
                "error": str(e),
                "answer": "",
                "citations": [],
                "_latency_ms": 120000,
            }

    def evaluate_item(
        self,
        benchmark_item: BenchmarkItem,
        api_result: Dict[str, Any],
    ) -> EvaluationResult:
        answer = api_result.get("answer", "")
        citations = api_result.get("citations", [])
        intent = api_result.get("intent", "")
        trace_id = api_result.get("trace_id", "")
        latency_ms = api_result.get("_latency_ms", 0)
        
        predicted_docs = []
        predicted_articles = []
        retrieved_doc_ids = []
        rerank_scores = []
        
        for citation in citations:
            doc_name = citation.get("doc_name", citation.get("source_name", ""))
            if doc_name and doc_name not in predicted_docs:
                predicted_docs.append(doc_name)
            article_no = citation.get("article_no", "")
            if article_no:
                predicted_articles.append(article_no)
        
        used_documents = api_result.get("used_documents", [])
        retrieved_doc_ids = used_documents if used_documents else predicted_docs
        
        keywords_hit = False
        if benchmark_item.expected_answer_keywords:
            for keyword in benchmark_item.expected_answer_keywords:
                if keyword.lower() in answer.lower():
                    keywords_hit = True
                    break
        
        is_correct = self._check_correctness(
            benchmark_item, predicted_docs, predicted_articles, answer
        )
        
        result = EvaluationResult(
            question_id=benchmark_item.question_id,
            question=benchmark_item.question,
            category=benchmark_item.category,
            expected_docs=benchmark_item.expected_docs,
            expected_articles=benchmark_item.expected_articles,
            expected_keywords=benchmark_item.expected_answer_keywords,
            should_reject=benchmark_item.should_reject,
            predicted_docs=predicted_docs,
            predicted_articles=predicted_articles,
            retrieved_doc_ids=retrieved_doc_ids,
            rerank_scores=rerank_scores,
            citations=citations,
            answer=answer,
            latency_ms=latency_ms,
            is_correct=is_correct,
            keywords_hit=keywords_hit,
            trace_id=trace_id,
        )
        
        return result

    def _check_correctness(
        self,
        benchmark_item: BenchmarkItem,
        predicted_docs: List[str],
        predicted_articles: List[str],
        answer: str,
    ) -> bool:
        if benchmark_item.should_reject:
            has_no_answer = not answer or "未检索到" in answer or "无法回答" in answer
            return has_no_answer
        
        doc_hit = False
        for expected_doc in benchmark_item.expected_docs:
            for pred_doc in predicted_docs:
                if expected_doc in pred_doc or pred_doc in expected_doc:
                    doc_hit = True
                    break
        
        keyword_hit = False
        for keyword in benchmark_item.expected_answer_keywords:
            if keyword.lower() in answer.lower():
                keyword_hit = True
                break
        
        return doc_hit or keyword_hit

    def run_benchmark(
        self,
        benchmark_path: str,
        category_filter: Optional[str] = None,
        use_ragas: bool = False,
    ) -> EvaluationReport:
        benchmark_items = self.load_benchmark(benchmark_path)
        
        if category_filter:
            benchmark_items = [
                item for item in benchmark_items if item.category == category_filter
            ]
        
        results: List[EvaluationResult] = []
        eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        for item in benchmark_items:
            api_result = self.call_query_api(item.question)
            result = self.evaluate_item(item, api_result)
            results.append(result)
        
        if use_ragas and self.ragas_evaluator:
            logger.info("Starting Ragas evaluation...")
            
            # Use batch processor if available
            if self.batch_processor:
                logger.info(f"Using batch processor with batch_size={self.ragas_config.batch_size}")
                ragas_scores = self.batch_processor.process_in_batches(
                    questions=[r.question for r in results],
                    answers=[r.answer for r in results],
                    contexts=[[c.get("excerpt", c.get("snippet", "")) for c in r.citations] for r in results],
                    evaluator=self.ragas_evaluator,
                )
            else:
                ragas_scores = self.ragas_evaluator.evaluate_batch(
                    questions=[r.question for r in results],
                    answers=[r.answer for r in results],
                    contexts=[[c.get("excerpt", c.get("snippet", "")) for c in r.citations] for r in results],
                )
            
            if "error" in ragas_scores:
                logger.warning(f"Ragas evaluation failed: {ragas_scores.get('error', 'Unknown error')}")
            else:
                logger.info("Ragas evaluation completed successfully")
                logger.info(f"Average faithfulness: {ragas_scores.get('avg_faithfulness', 0):.2f}")
                logger.info(f"Average answer_relevancy: {ragas_scores.get('avg_answer_relevancy', 0):.2f}")
                logger.info(f"Average context_precision: {ragas_scores.get('avg_context_precision', 0):.2f}")
                
                for i, result in enumerate(results):
                    result.faithfulness_score = ragas_scores.get("faithfulness", {}).get(i, 0.0)
                    result.answer_relevancy_score = ragas_scores.get("answer_relevancy", {}).get(i, 0.0)
                    result.context_precision_score = ragas_scores.get("context_precision", {}).get(i, 0.0)
        
        self.results = results
        report = self._generate_report(results, eval_id, benchmark_path)
        
        return report

    def _generate_report(
        self,
        results: List[EvaluationResult],
        eval_id: str,
        benchmark_path: str,
    ) -> EvaluationReport:
        metrics_report = compute_all_metrics(results)
        threshold_checks = check_threshold(metrics_report)
        overall = overall_pass(threshold_checks)
        
        failed_questions = [
            r.question_id for r in results if not r.is_correct
        ]
        
        metrics_dict = {
            "recall@3": metrics_report.recall_at_3,
            "recall@5": metrics_report.recall_at_5,
            "precision@k": metrics_report.precision_at_k,
            "hit_rate": metrics_report.hit_rate,
            "avg_score": metrics_report.avg_score,
            "ood_rate": metrics_report.ood_rate,
            "citation_rate": metrics_report.citation_rate,
            "citation_accuracy": metrics_report.citation_accuracy,
            "formal_doc_priority": metrics_report.formal_doc_priority,
            "draft_misuse_rate": metrics_report.draft_misuse_rate,
            "avg_latency_ms": metrics_report.avg_latency_ms,
            "p99_latency_ms": metrics_report.p99_latency_ms,
            "pass_rate": metrics_report.pass_count / metrics_report.total_questions if metrics_report.total_questions > 0 else 0,
        }
        
        if metrics_report.faithfulness is not None:
            metrics_dict["faithfulness"] = metrics_report.faithfulness
        if metrics_report.answer_relevancy is not None:
            metrics_dict["answer_relevancy"] = metrics_report.answer_relevancy
        if metrics_report.context_precision is not None:
            metrics_dict["context_precision"] = metrics_report.context_precision
        if metrics_report.flow_complete_rate is not None:
            metrics_dict["flow_complete_rate"] = metrics_report.flow_complete_rate
        if metrics_report.context_continuation_rate is not None:
            metrics_dict["context_continuation_rate"] = metrics_report.context_continuation_rate
        if metrics_report.rejection_correct_rate is not None:
            metrics_dict["rejection_correct_rate"] = metrics_report.rejection_correct_rate
        
        return EvaluationReport(
            eval_id=eval_id,
            timestamp=datetime.now().isoformat(),
            total_questions=len(results),
            metrics=metrics_dict,
            threshold_check=threshold_checks,
            overall_pass=overall,
            failed_questions=failed_questions,
            results=results,
            benchmark_version=Path(benchmark_path).stem,
        )

    def compare_reports(
        self,
        baseline: EvaluationReport,
        current: EvaluationReport,
    ) -> Dict[str, Any]:
        comparison = {
            "baseline_eval_id": baseline.eval_id,
            "current_eval_id": current.eval_id,
            "metric_changes": {},
            "overall_improvement": True,
            "recommendation": "",
        }
        
        improved_count = 0
        degraded_count = 0
        
        for metric, baseline_value in baseline.metrics.items():
            current_value = current.metrics.get(metric, 0)
            delta = current_value - baseline_value
            
            is_improvement = self._is_metric_improvement(metric, delta)
            
            comparison["metric_changes"][metric] = {
                "baseline": baseline_value,
                "current": current_value,
                "delta": round(delta, 4),
                "improved": is_improvement,
            }
            
            if is_improvement:
                improved_count += 1
            elif delta != 0:
                degraded_count += 1
        
        comparison["overall_improvement"] = improved_count > degraded_count
        
        if comparison["overall_improvement"]:
            comparison["recommendation"] = "当前版本优于基准版本，建议采纳"
        elif degraded_count > improved_count:
            comparison["recommendation"] = "当前版本劣于基准版本，不建议采纳"
        else:
            comparison["recommendation"] = "当前版本与基准版本持平，需人工判断"
        
        return comparison

    def _is_metric_improvement(self, metric: str, delta: float) -> bool:
        negative_metrics = ["ood_rate", "draft_misuse_rate", "avg_latency_ms", "p99_latency_ms"]
        if metric in negative_metrics:
            return delta < 0
        return delta > 0

    def save_to_database(self, report: EvaluationReport) -> None:
        if not self.session_factory:
            return
        
        with self.session_factory() as db:
            from app.db.models.evaluation_record import EvaluationRecord
            from app.db.models.evaluation_session import EvaluationSession
            
            session = EvaluationSession(
                eval_id=report.eval_id,
                benchmark_version=report.benchmark_version,
                total_questions=report.total_questions,
                pass_count=len([r for r in report.results if r.is_correct]),
                overall_pass=report.overall_pass,
                metrics_json=json.dumps(report.metrics),
                git_commit=report.git_commit,
                config_snapshot=report.config_snapshot,
            )
            db.add(session)
            
            for result in report.results:
                record = EvaluationRecord(
                    question=result.question,
                    question_id=result.question_id,
                    expected_doc=result.expected_docs[0] if result.expected_docs else None,
                    expected_article=result.expected_articles[0] if result.expected_articles else None,
                    predicted_doc=result.predicted_docs[0] if result.predicted_docs else None,
                    predicted_article=result.predicted_articles[0] if result.predicted_articles else None,
                    is_correct=result.is_correct,
                    latency_ms=result.latency_ms,
                    trace_id=result.trace_id,
                    category=result.category,
                    eval_session_id=report.eval_id,
                    benchmark_version=report.benchmark_version,
                    expected_keywords_hit=result.keywords_hit,
                    answer_text=result.answer,
                    llm_faithfulness_score=result.faithfulness_score,
                    llm_answer_relevancy_score=result.answer_relevancy_score,
                    llm_context_precision_score=result.context_precision_score,
                )
                db.add(record)
            
            db.commit()
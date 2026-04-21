from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class EvaluationResult:
    question_id: str
    question: str
    category: str
    expected_docs: List[str] = field(default_factory=list)
    expected_articles: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    should_reject: bool = False
    predicted_docs: List[str] = field(default_factory=list)
    predicted_articles: List[str] = field(default_factory=list)
    retrieved_doc_ids: List[str] = field(default_factory=list)
    rerank_scores: List[float] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    latency_ms: int = 0
    is_correct: bool = False
    keywords_hit: bool = False
    trace_id: str = ""
    faithfulness_score: float | None = None
    answer_relevancy_score: float | None = None
    context_precision_score: float | None = None


@dataclass
class MetricsReport:
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_k: float = 0.0
    hit_rate: float = 0.0
    avg_score: float = 0.0
    ood_rate: float = 0.0
    citation_rate: float = 0.0
    citation_accuracy: float = 0.0
    formal_doc_priority: float = 0.0
    draft_misuse_rate: float = 0.0
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    flow_complete_rate: float | None = None
    context_continuation_rate: float | None = None
    rejection_correct_rate: float | None = None
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    total_questions: int = 0
    pass_count: int = 0


THRESHOLDS = {
    "recall@3": {"target": 0.85, "pass": lambda v: v >= 0.85},
    "recall@5": {"target": 0.90, "pass": lambda v: v >= 0.90},
    "precision@k": {"target": 0.80, "pass": lambda v: v >= 0.80},
    "hit_rate": {"target": 0.90, "pass": lambda v: v >= 0.90},
    "avg_score": {"target": 0.70, "pass": lambda v: v >= 0.70},
    "ood_rate": {"target": 0.10, "pass": lambda v: v <= 0.10},
    "citation_rate": {"target": 0.95, "pass": lambda v: v >= 0.95},
    "citation_accuracy": {"target": 0.90, "pass": lambda v: v >= 0.90},
    "formal_doc_priority": {"target": 0.95, "pass": lambda v: v >= 0.95},
    "draft_misuse_rate": {"target": 0.05, "pass": lambda v: v <= 0.05},
    "faithfulness": {"target": 0.85, "pass": lambda v: v is not None and v >= 0.85},
    "answer_relevancy": {"target": 0.85, "pass": lambda v: v is not None and v >= 0.85},
    "context_precision": {"target": 0.80, "pass": lambda v: v is not None and v >= 0.80},
    "flow_complete_rate": {"target": 0.85, "pass": lambda v: v is None or v >= 0.85},
    "context_continuation_rate": {"target": 0.80, "pass": lambda v: v is None or v >= 0.80},
    "rejection_correct_rate": {"target": 0.85, "pass": lambda v: v is None or v >= 0.85},
    "avg_latency_ms": {"target": 8000, "pass": lambda v: v <= 8000},
    "p99_latency_ms": {"target": 15000, "pass": lambda v: v <= 15000},
}


def recall_at_k(results: List[EvaluationResult], k: int = 3) -> float:
    if not results:
        return 0.0
    hits = 0
    for r in results:
        if not r.expected_docs:
            continue
        retrieved = r.retrieved_doc_ids[:k] if r.retrieved_doc_ids else []
        predicted = r.predicted_docs[:k] if r.predicted_docs else []
        combined = set(retrieved + predicted)
        for expected in r.expected_docs:
            if expected in combined:
                hits += 1
                break
    total = len([r for r in results if r.expected_docs])
    return hits / total if total > 0 else 0.0


def precision_at_k(results: List[EvaluationResult], k: int = 3) -> float:
    if not results:
        return 0.0
    total_relevant = 0
    total_retrieved = 0
    for r in results:
        retrieved = r.retrieved_doc_ids[:k] if r.retrieved_doc_ids else []
        predicted = r.predicted_docs[:k] if r.predicted_docs else []
        combined = list(set(retrieved + predicted))[:k]
        total_retrieved += len(combined)
        for doc in combined:
            if doc in r.expected_docs:
                total_relevant += 1
    return total_relevant / total_retrieved if total_retrieved > 0 else 0.0


def hit_rate(results: List[EvaluationResult]) -> float:
    if not results:
        return 0.0
    hits = 0
    for r in results:
        if not r.expected_docs:
            continue
        combined = set(r.retrieved_doc_ids + r.predicted_docs)
        for expected in r.expected_docs:
            if expected in combined:
                hits += 1
                break
    total = len([r for r in results if r.expected_docs])
    return hits / total if total > 0 else 0.0


def avg_score(results: List[EvaluationResult]) -> float:
    if not results:
        return 0.0
    scores = []
    for r in results:
        if r.rerank_scores:
            scores.extend(r.rerank_scores)
    return statistics.mean(scores) if scores else 0.0


def ood_rate(results: List[EvaluationResult]) -> float:
    if not results:
        return 0.0
    ood_count = 0
    for r in results:
        avg_retrieval_score = statistics.mean(r.rerank_scores) if r.rerank_scores else 0.0
        if avg_retrieval_score < 0.3 and not r.predicted_docs:
            ood_count += 1
    return ood_count / len(results)


def citation_rate(results: List[EvaluationResult]) -> float:
    if not results:
        return 0.0
    with_citations = len([r for r in results if r.citations and len(r.citations) > 0])
    return with_citations / len(results)


def citation_accuracy(results: List[EvaluationResult]) -> float:
    if not results:
        return 0.0
    correct_count = 0
    total_with_expected = 0
    for r in results:
        if not r.expected_articles:
            continue
        total_with_expected += 1
        for citation in r.citations:
            article_no = citation.get("article_no", "")
            title_path = citation.get("title_path", "")
            for expected_article in r.expected_articles:
                if expected_article in article_no or expected_article in title_path:
                    correct_count += 1
                    break
    return correct_count / total_with_expected if total_with_expected > 0 else 0.0


def formal_doc_priority(results: List[EvaluationResult]) -> float:
    if not results:
        return 0.0
    formal_count = 0
    total_citations = 0
    for r in results:
        for citation in r.citations:
            status = citation.get("status", "formal")
            if status == "formal":
                formal_count += 1
            total_citations += 1
    return formal_count / total_citations if total_citations > 0 else 1.0


def draft_misuse_rate(results: List[EvaluationResult]) -> float:
    if not results:
        return 0.0
    draft_count = 0
    total_citations = 0
    for r in results:
        for citation in r.citations:
            status = citation.get("status", "formal")
            if status == "draft":
                draft_count += 1
            total_citations += 1
    return draft_count / total_citations if total_citations > 0 else 0.0


def avg_latency(results: List[EvaluationResult]) -> float:
    if not results:
        return 0.0
    latencies = [r.latency_ms for r in results]
    return statistics.mean(latencies) if latencies else 0.0


def p99_latency(results: List[EvaluationResult]) -> float:
    if not results:
        return 0.0
    latencies = sorted([r.latency_ms for r in results])
    if len(latencies) < 2:
        return latencies[0] if latencies else 0.0
    idx = int(len(latencies) * 0.99)
    return latencies[min(idx, len(latencies) - 1)]


def flow_complete_rate(results: List[EvaluationResult]) -> float | None:
    flow_results = [r for r in results if r.category == "flow_qa"]
    if not flow_results:
        return None
    complete_count = len([r for r in flow_results if r.keywords_hit and r.is_correct])
    return complete_count / len(flow_results)


def context_continuation_rate(results: List[EvaluationResult]) -> float | None:
    context_results = [r for r in results if r.category == "context_qa"]
    if not context_results:
        return None
    correct_count = len([r for r in context_results if r.is_correct])
    return correct_count / len(context_results)


def rejection_correct_rate(results: List[EvaluationResult]) -> float | None:
    rejection_results = [r for r in results if r.should_reject]
    if not rejection_results:
        return None
    correct_rejections = 0
    for r in rejection_results:
        has_no_answer = not r.answer or "未检索到" in r.answer or "无法回答" in r.answer or "不在知识库" in r.answer
        if has_no_answer:
            correct_rejections += 1
    return correct_rejections / len(rejection_results)


def keywords_hit_rate(results: List[EvaluationResult]) -> float:
    if not results:
        return 0.0
    hits = len([r for r in results if r.keywords_hit])
    return hits / len(results)


def compute_all_metrics(results: List[EvaluationResult]) -> MetricsReport:
    report = MetricsReport(
        recall_at_3=recall_at_k(results, k=3),
        recall_at_5=recall_at_k(results, k=5),
        precision_at_k=precision_at_k(results, k=3),
        hit_rate=hit_rate(results),
        avg_score=avg_score(results),
        ood_rate=ood_rate(results),
        citation_rate=citation_rate(results),
        citation_accuracy=citation_accuracy(results),
        formal_doc_priority=formal_doc_priority(results),
        draft_misuse_rate=draft_misuse_rate(results),
        avg_latency_ms=avg_latency(results),
        p99_latency_ms=p99_latency(results),
        flow_complete_rate=flow_complete_rate(results),
        context_continuation_rate=context_continuation_rate(results),
        rejection_correct_rate=rejection_correct_rate(results),
        total_questions=len(results),
        pass_count=len([r for r in results if r.is_correct]),
    )
    
    llm_scores = [r for r in results if r.faithfulness_score is not None]
    if llm_scores:
        report.faithfulness = statistics.mean([r.faithfulness_score for r in llm_scores if r.faithfulness_score])
        report.answer_relevancy = statistics.mean([r.answer_relevancy_score for r in llm_scores if r.answer_relevancy_score])
        report.context_precision = statistics.mean([r.context_precision_score for r in llm_scores if r.context_precision_score])
    
    return report


def check_threshold(report: MetricsReport) -> Dict[str, Dict[str, Any]]:
    checks = {}
    metric_values = {
        "recall@3": report.recall_at_3,
        "recall@5": report.recall_at_5,
        "precision@k": report.precision_at_k,
        "hit_rate": report.hit_rate,
        "avg_score": report.avg_score,
        "ood_rate": report.ood_rate,
        "citation_rate": report.citation_rate,
        "citation_accuracy": report.citation_accuracy,
        "formal_doc_priority": report.formal_doc_priority,
        "draft_misuse_rate": report.draft_misuse_rate,
        "faithfulness": report.faithfulness,
        "answer_relevancy": report.answer_relevancy,
        "context_precision": report.context_precision,
        "flow_complete_rate": report.flow_complete_rate,
        "context_continuation_rate": report.context_continuation_rate,
        "rejection_correct_rate": report.rejection_correct_rate,
        "avg_latency_ms": report.avg_latency_ms,
        "p99_latency_ms": report.p99_latency_ms,
    }
    
    for metric, value in metric_values.items():
        threshold_info = THRESHOLDS.get(metric, {})
        target = threshold_info.get("target", 0)
        pass_func = threshold_info.get("pass", lambda v: True)
        checks[metric] = {
            "target": target,
            "actual": value,
            "pass": pass_func(value),
        }
    
    return checks


def overall_pass(checks: Dict[str, Dict[str, Any]]) -> bool:
    for metric, info in checks.items():
        if not info.get("pass", True):
            return False
    return True
from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import Callable

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.db.models.trace_record import TraceRecord
from app.db.models.evaluation_record import EvaluationRecord
from evaluation.ragas_evaluator import get_ragas_evaluator

logger = logging.getLogger(__name__)


class EvaluationService:
    def __init__(
        self,
        session_factory: Callable[[], Session],
        use_mock: bool = False,
    ):
        self.session_factory = session_factory
        self.use_mock = use_mock
        self.evaluator = get_ragas_evaluator(use_mock=use_mock)

    def get_pending_traces(self, limit: int = 50) -> list[TraceRecord]:
        """
        Get traces that have contexts and answer but not yet evaluated.
        
        Returns traces with retrieved_doc_texts and answer_text populated
        but not yet in evaluation_record.
        """
        with self.session_factory() as db:
            evaluated_trace_ids = db.scalars(
                select(EvaluationRecord.trace_id)
            ).all()
            
            traces = db.scalars(
                select(TraceRecord)
                .where(TraceRecord.retrieved_doc_texts.isnot(None))
                .where(TraceRecord.answer_text.isnot(None))
                .where(TraceRecord.success == True)
                .where(~TraceRecord.trace_id.in_(evaluated_trace_ids))
                .order_by(TraceRecord.created_at.desc())
                .limit(limit)
            ).all()
            
            return list(traces)

    def evaluate_traces(self, traces: list[TraceRecord]) -> list[EvaluationRecord]:
        """
        Evaluate a batch of traces using RAGAS.
        
        Args:
            traces: List of TraceRecord to evaluate
            
        Returns:
            List of EvaluationRecord with RAGAS scores
        """
        if not traces:
            return []
        
        import json
        
        questions = [t.raw_query for t in traces]
        answers = [t.answer_text for t in traces]
        contexts = [json.loads(t.retrieved_doc_texts or "[]") for t in traces]
        
        result = self.evaluator.evaluate_batch(questions, answers, contexts)
        
        records = []
        with self.session_factory() as db:
            for i, trace in enumerate(traces):
                faithfulness = result.get("faithfulness", {}).get(i, 0)
                answer_relevancy = result.get("answer_relevancy", {}).get(i, 0)
                context_precision = result.get("context_precision", {}).get(i, 0)
                
                record = EvaluationRecord(
                    question=trace.raw_query,
                    trace_id=trace.trace_id,
                    answer_text=trace.answer_text,
                    llm_faithfulness_score=faithfulness,
                    llm_answer_relevancy_score=answer_relevancy,
                    llm_context_precision_score=context_precision,
                    eval_session_id=f"eval_{uuid.uuid4().hex[:8]}",
                    created_at=dt.datetime.utcnow(),
                )
                db.add(record)
                records.append(record)
            
            db.commit()
        
        logger.info(f"Evaluated {len(records)} traces with RAGAS")
        return records

    def run_batch_evaluation(self, batch_size: int = 20) -> dict:
        """
        Run batch evaluation on pending traces.
        
        Args:
            batch_size: Number of traces to evaluate in one batch
            
        Returns:
            Summary of evaluation results
        """
        traces = self.get_pending_traces(limit=batch_size)
        
        if not traces:
            return {
                "evaluated_count": 0,
                "message": "No pending traces to evaluate",
            }
        
        records = self.evaluate_traces(traces)
        
        if not records:
            return {
                "evaluated_count": 0,
                "message": "Evaluation failed",
            }
        
        avg_faithfulness = sum(r.llm_faithfulness_score or 0 for r in records) / len(records)
        avg_answer_relevancy = sum(r.llm_answer_relevancy_score or 0 for r in records) / len(records)
        avg_context_precision = sum(r.llm_context_precision_score or 0 for r in records) / len(records)
        
        return {
            "evaluated_count": len(records),
            "avg_faithfulness": round(avg_faithfulness, 4),
            "avg_answer_relevancy": round(avg_answer_relevancy, 4),
            "avg_context_precision": round(avg_context_precision, 4),
        }

    def get_recent_evaluations(self, limit: int = 50) -> list[dict]:
        """
        Get recent evaluation records.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of evaluation records with scores
        """
        with self.session_factory() as db:
            records = db.scalars(
                select(EvaluationRecord)
                .where(EvaluationRecord.llm_faithfulness_score.isnot(None))
                .order_by(EvaluationRecord.created_at.desc())
                .limit(limit)
            ).all()
            
            return [
                {
                    "trace_id": r.trace_id,
                    "question": r.question,
                    "answer": r.answer_text,
                    "faithfulness": r.llm_faithfulness_score,
                    "answer_relevancy": r.llm_answer_relevancy_score,
                    "context_precision": r.llm_context_precision_score,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in records
            ]

    def get_evaluation_summary(self, hours: int = 24) -> dict:
        """
        Get evaluation metrics summary.
        
        Args:
            hours: Time window in hours
            
        Returns:
            Summary with average scores
        """
        with self.session_factory() as db:
            cutoff = dt.datetime.utcnow() - dt.timedelta(hours=hours)
            
            result = db.execute(
                select(
                    func.count(EvaluationRecord.id).label("total"),
                    func.avg(EvaluationRecord.llm_faithfulness_score).label("avg_faithfulness"),
                    func.avg(EvaluationRecord.llm_answer_relevancy_score).label("avg_answer_relevancy"),
                    func.avg(EvaluationRecord.llm_context_precision_score).label("avg_context_precision"),
                )
                .where(EvaluationRecord.created_at >= cutoff)
                .where(EvaluationRecord.llm_faithfulness_score.isnot(None))
            ).first()
            
            if result is None or result.total == 0:
                return {
                    "total_records": 0,
                    "avg_faithfulness": 0,
                    "avg_answer_relevancy": 0,
                    "avg_context_precision": 0,
                    "period_hours": hours,
                }
            
            return {
                "total_records": result.total,
                "avg_faithfulness": round(result.avg_faithfulness or 0, 4),
                "avg_answer_relevancy": round(result.avg_answer_relevancy or 0, 4),
                "avg_context_precision": round(result.avg_context_precision or 0, 4),
                "period_hours": hours,
            }

    def get_hourly_metrics(self, hours: int = 24) -> list[dict]:
        """
        Get hourly evaluation metrics.
        
        Args:
            hours: Time window in hours
            
        Returns:
            List of hourly metrics
        """
        with self.session_factory() as db:
            cutoff = dt.datetime.utcnow() - dt.timedelta(hours=hours)
            
            records = db.scalars(
                select(EvaluationRecord)
                .where(EvaluationRecord.created_at >= cutoff)
                .where(EvaluationRecord.llm_faithfulness_score.isnot(None))
                .order_by(EvaluationRecord.created_at)
            ).all()
            
            hourly_data = {}
            for r in records:
                hour_key = r.created_at.strftime("%Y-%m-%d %H:00")
                if hour_key not in hourly_data:
                    hourly_data[hour_key] = {
                        "faithfulness": [],
                        "answer_relevancy": [],
                        "context_precision": [],
                    }
                hourly_data[hour_key]["faithfulness"].append(r.llm_faithfulness_score or 0)
                hourly_data[hour_key]["answer_relevancy"].append(r.llm_answer_relevancy_score or 0)
                hourly_data[hour_key]["context_precision"].append(r.llm_context_precision_score or 0)
            
            result = []
            for hour, scores in sorted(hourly_data.items()):
                result.append({
                    "hour": hour,
                    "faithfulness": round(sum(scores["faithfulness"]) / len(scores["faithfulness"]), 4),
                    "answer_relevancy": round(sum(scores["answer_relevancy"]) / len(scores["answer_relevancy"]), 4),
                    "context_precision": round(sum(scores["context_precision"]) / len(scores["context_precision"]), 4),
                    "count": len(scores["faithfulness"]),
                })
            
            return result
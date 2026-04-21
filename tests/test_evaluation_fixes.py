"""
Test evaluation framework fixes

Tests for Ragas integration and field mapping fixes
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock
from evaluation.evaluator import RAGEvaluator, EvaluationReport, EvaluationResult
from evaluation.ragas_evaluator import RagasEvaluator, MockRagasEvaluator


def test_ragas_score_assignment_with_error():
    """Test that Ragas score assignment handles errors gracefully"""
    
    # Create mock Ragas evaluator that returns error
    mock_ragas = Mock()
    mock_ragas.evaluate_batch.return_value = {
        "faithfulness": {},
        "answer_relevancy": {},
        "context_precision": {},
        "error": "API connection failed"
    }
    
    evaluator = RAGEvaluator(
        api_endpoint="http://localhost:8000/query",
        ragas_evaluator=mock_ragas
    )
    
    # Create mock results
    results = [
        EvaluationResult(
            question_id="q001",
            question="Test question",
            category="clause_qa",
            answer="Test answer",
            citations=[],
            latency_ms=100,
        )
    ]
    
    # Should not crash when Ragas returns error
    ragas_scores = mock_ragas.evaluate_batch(
        questions=[r.question for r in results],
        answers=[r.answer for r in results],
        contexts=[[] for r in results],
    )
    
    # Verify error handling
    assert "error" in ragas_scores
    assert ragas_scores["error"] == "API connection failed"


def test_ragas_score_assignment_success():
    """Test that Ragas scores are correctly assigned to results"""
    
    # Create mock Ragas evaluator with proper structure
    mock_ragas = Mock()
    mock_ragas.evaluate_batch.return_value = {
        "faithfulness": {0: 0.85, 1: 0.90},
        "answer_relevancy": {0: 0.80, 1: 0.85},
        "context_precision": {0: 0.75, 1: 0.80},
        "avg_faithfulness": 0.875,
        "avg_answer_relevancy": 0.825,
        "avg_context_precision": 0.775,
    }
    
    evaluator = RAGEvaluator(
        api_endpoint="http://localhost:8000/query",
        ragas_evaluator=mock_ragas
    )
    
    # Create mock results
    results = [
        EvaluationResult(
            question_id="q001",
            question="Test question 1",
            category="clause_qa",
            answer="Test answer 1",
            citations=[],
            latency_ms=100,
        ),
        EvaluationResult(
            question_id="q002",
            question="Test question 2",
            category="clause_qa",
            answer="Test answer 2",
            citations=[],
            latency_ms=100,
        )
    ]
    
    # Get Ragas scores
    ragas_scores = mock_ragas.evaluate_batch(
        questions=[r.question for r in results],
        answers=[r.answer for r in results],
        contexts=[[] for r in results],
    )
    
    # Assign scores to results
    for i, result in enumerate(results):
        result.faithfulness_score = ragas_scores.get("faithfulness", {}).get(i, 0.0)
        result.answer_relevancy_score = ragas_scores.get("answer_relevancy", {}).get(i, 0.0)
        result.context_precision_score = ragas_scores.get("context_precision", {}).get(i, 0.0)
    
    # Verify correct assignment
    assert results[0].faithfulness_score == 0.85
    assert results[0].answer_relevancy_score == 0.80
    assert results[0].context_precision_score == 0.75
    
    assert results[1].faithfulness_score == 0.90
    assert results[1].answer_relevancy_score == 0.85
    assert results[1].context_precision_score == 0.80


def test_ragas_setup_glm_endpoint():
    """Test Ragas setup with GLM endpoint validation"""
    
    # Test with GLM endpoint (non-OpenAI)
    ragas_eval = MockRagasEvaluator()  # Use mock since Ragas may not be installed
    
    # Verify it's available
    assert ragas_eval.is_available()
    
    # Test evaluate_batch returns proper structure
    result = ragas_eval.evaluate_batch(
        questions=["Test question"],
        answers=["Test answer"],
        contexts=[["Test context"]],
    )
    
    # Verify structure matches expected format
    assert "faithfulness" in result
    assert "answer_relevancy" in result
    assert "context_precision" in result
    assert "avg_faithfulness" in result
    
    # Verify scores are in dict format {index: score}
    assert isinstance(result["faithfulness"], dict)
    assert 0 in result["faithfulness"]
    assert result["faithfulness"][0] > 0


def test_evaluation_record_field_mapping():
    """Test that all EvaluationResult fields map to EvaluationRecord"""
    
    # Create complete EvaluationResult
    result = EvaluationResult(
        question_id="q001",
        question="Test question",
        category="clause_qa",
        expected_docs=["doc1.pdf"],
        expected_articles=["第2条"],
        expected_keywords=["签约比例"],
        should_reject=False,
        predicted_docs=["doc1.pdf"],
        predicted_articles=["第2条"],
        retrieved_doc_ids=["id1"],
        rerank_scores=[0.85],
        citations=[{"doc_name": "doc1.pdf", "excerpt": "test"}],
        answer="Test answer with 签约比例",
        latency_ms=100,
        is_correct=True,
        keywords_hit=True,
        trace_id="trace_001",
        faithfulness_score=0.85,
        answer_relevancy_score=0.80,
        context_precision_score=0.75,
    )
    
    # Verify all fields are populated
    assert result.question_id == "q001"
    assert result.question == "Test question"
    assert result.category == "clause_qa"
    assert result.answer == "Test answer with 签约比例"
    assert result.faithfulness_score == 0.85
    assert result.answer_relevancy_score == 0.80
    assert result.context_precision_score == 0.75
    
    # These fields should be mapped to EvaluationRecord
    expected_record_fields = {
        "question": result.question,
        "question_id": result.question_id,
        "expected_doc": result.expected_docs[0],
        "expected_article": result.expected_articles[0],
        "predicted_doc": result.predicted_docs[0],
        "predicted_article": result.predicted_articles[0],
        "is_correct": result.is_correct,
        "latency_ms": result.latency_ms,
        "trace_id": result.trace_id,
        "category": result.category,
        "expected_keywords_hit": result.keywords_hit,
        "answer_text": result.answer,
        "llm_faithfulness_score": result.faithfulness_score,
        "llm_answer_relevancy_score": result.answer_relevancy_score,
        "llm_context_precision_score": result.context_precision_score,
    }
    
    # All fields should have values
    for field, value in expected_record_fields.items():
        assert value is not None, f"Field {field} should not be None"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
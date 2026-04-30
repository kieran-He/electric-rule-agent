"""Integration tests: Verify evaluation full workflow"""

import json
from pathlib import Path

import pytest


BENCHMARK_PATH = Path("evaluation/benchmark.json")


def test_benchmark_file_exists():
    assert BENCHMARK_PATH.exists()


def test_benchmark_structure():
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        benchmark = json.load(f)
    
    assert "version" in benchmark
    assert benchmark["version"] == "v3.0_multi_province"
    assert "questions" in benchmark
    assert len(benchmark["questions"]) == 100
    assert "province_scope" in benchmark
    assert set(benchmark["province_scope"]) == {"SN", "SX", "GS", "AH", "SD"}
    
    for q in benchmark["questions"]:
        assert "question_id" in q
        assert "question" in q
        assert "category" in q
        assert "province" in q
        assert "expected_docs" in q
        assert "expected_answer_keywords" in q
        assert "should_reject" in q
        assert "expected_intent" in q


def test_benchmark_distribution():
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        benchmark = json.load(f)
    
    dist = benchmark.get("distribution", {})
    assert dist.get("clause_qa") == 40
    assert dist.get("flow_qa") == 20
    assert dist.get("compare_qa") == 15
    assert dist.get("settlement_qa") == 15
    assert dist.get("rejection") == 10


def test_benchmark_expected_docs_not_empty():
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        benchmark = json.load(f)
    
    for q in benchmark["questions"]:
        if q["category"] != "rejection":
            assert len(q["expected_docs"]) > 0, f"Question {q['question_id']} missing expected_docs"


def test_benchmark_rejection_questions():
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        benchmark = json.load(f)
    
    rejection_questions = [q for q in benchmark["questions"] if q["category"] == "rejection"]
    assert len(rejection_questions) == 10
    
    for q in rejection_questions:
        assert q["should_reject"] == True
        assert q["expected_docs"] == []
        assert q["province"] == "N/A"


def test_benchmark_province_distribution():
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        benchmark = json.load(f)
    
    province_dist = benchmark.get("province_distribution", {})
    assert "SN" in province_dist
    assert "SX" in province_dist
    assert "GS" in province_dist
    assert "AH" in province_dist
    assert "SD" in province_dist
    assert "MULTI" in province_dist
    assert "N/A" in province_dist
    
    multi_questions = [q for q in benchmark["questions"] if q["province"] == "MULTI"]
    assert all(q["category"] == "compare_qa" for q in multi_questions)
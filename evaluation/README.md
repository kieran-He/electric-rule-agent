# RAG Evaluation System

## Overview

Complete evaluation framework for assessing RAG system quality after each code change, supporting A/B comparison and continuous optimization.

## Architecture

```
evaluation/
├── benchmark_generator.py  # Generate test questions from knowledge base
├── metrics.py              # Calculate evaluation metrics
├── evaluator.py            # Core evaluation logic
├── ragas_evaluator.py      # LLM quality evaluation (Ragas)
├── report_generator.py     # Generate JSON/HTML reports
├── run_eval.py             # CLI entry point
└── benchmark.json          # 100 test questions
```

## Quick Start

### 1. Generate Benchmark

Generate 100 test questions from knowledge base documents:

```bash
python evaluation/run_eval.py generate --docs-path data/docs/SN --output evaluation/benchmark.json --count 100
```

### 2. Run Evaluation

Run full evaluation against `/query` API:

```bash
python evaluation/run_eval.py run --benchmark evaluation/benchmark.json
```

### 3. View Reports

Reports are saved to `evaluation/reports/`:
- `eval_YYYYMMDD_HHMMSS_HASH.json` - Detailed JSON report
- `eval_YYYYMMDD_HHMMSS_HASH.html` - Visual HTML report

## CLI Commands

### Run Evaluation

```bash
# Basic evaluation
python evaluation/run_eval.py run --benchmark evaluation/benchmark.json

# Filter by category
python evaluation/run_eval.py run --benchmark evaluation/benchmark.json --category clause_qa

# Enable Ragas LLM evaluation
python evaluation/run_eval.py run --benchmark evaluation/benchmark.json --ragas

# Save to database
python evaluation/run_eval.py run --benchmark evaluation/benchmark.json --save-db

# Custom API endpoint
python evaluation/run_eval.py run --benchmark evaluation/benchmark.json --api-endpoint http://localhost:8000/query
```

### Generate Benchmark

```bash
# Default: 100 questions from data/docs/SN
python evaluation/run_eval.py generate

# Custom path
python evaluation/run_eval.py generate --docs-path path/to/docs --output custom_benchmark.json --count 50
```

### Compare Evaluations (A/B Testing)

```bash
# Compare two evaluation results
python evaluation/run_eval.py compare eval_20260421_001 eval_20260421_002

# Output comparison report to evaluation/reports/
```

### Set Baseline

```bash
# Mark an evaluation as baseline
python evaluation/run_eval.py set-baseline eval_20260421_001
```

## Metrics (17 Indicators)

### Retrieval Metrics
- **recall@k**: Top-k retrieval hit rate (target ≥85%)
- **precision@k**: Top-k retrieval relevance (target ≥80%)
- **hit_rate**: At least one correct document hit (target ≥90%)
- **avg_score**: Vector similarity average (target ≥0.70)
- **OOD_rate**: Out-of-distribution query rate (target ≤10%)

### Answer Quality Metrics
- **citation_rate**: Answer with citations rate (target ≥95%)
- **citation_accuracy**: Article/title path accuracy (target ≥90%)
- **formal_doc_priority**: Formal document priority hit rate (target ≥95%)
- **draft_misuse_rate**: Draft document misuse rate (target ≤5%)
- **faithfulness**: Answer faithfulness to documents (Ragas, target ≥85%)
- **answer_relevancy**: Answer relevance to question (Ragas, target ≥85%)
- **context_precision**: Context relevance to question (Ragas, target ≥80%)

### Flow & Context Metrics
- **flow_complete_rate**: Flow question completeness (target ≥85%)
- **context_continuation_rate**: Multi-turn context accuracy (target ≥80%)
- **rejection_correct_rate**: Correct rejection rate (target ≥85%)

### Performance Metrics
- **avg_latency_ms**: Average response time (target ≤8000ms)
- **p99_latency_ms**: 99th percentile latency (target ≤15000ms)

## Benchmark Structure

100 questions divided into 5 categories:

- **Clause QA** (40): Based on knowledge base, covering various rules
- **Flow QA** (20): Process-based questions (storage, VPP, retail company)
- **Compare QA** (15): Cross-province or cross-entity comparisons
- **Settlement QA** (15): Settlement and metering rules
- **Rejection** (10): Knowledge-out-of-scope questions

Question distribution:
- Length: Short (≤15) 30%, Medium (15-30) 50%, Long (>30) 20%
- Scope: Macro summary 20%, Micro clause 60%, Cross-doc 20%

## Database Models

### EvaluationSession

Stores overall evaluation session information:

```python
- eval_id: Unique evaluation ID
- benchmark_version: Benchmark version
- total_questions: Total question count
- pass_count: Passed question count
- overall_pass: Overall pass/fail status
- metrics_json: All metrics in JSON format
- git_commit: Code version
- is_baseline: Whether this is baseline
```

### EvaluationRecord

Stores individual question evaluation results:

```python
- question_id: Question ID
- question: Question text
- category: Question category
- expected_docs: Expected documents
- expected_articles: Expected articles
- predicted_docs: Predicted documents
- predicted_articles: Predicted articles
- is_correct: Whether prediction is correct
- keywords_hit: Whether keywords hit
- latency_ms: Response latency
- llm_faithfulness_score: Ragas faithfulness score
- llm_answer_relevancy_score: Ragas relevancy score
```

## Report Format

### JSON Report

```json
{
  "eval_id": "eval_20260421_140000_a1b2c3d4",
  "timestamp": "2026-04-21T14:00:00",
  "total_questions": 100,
  "metrics": {
    "recall@3": 0.87,
    "citation_rate": 0.96,
    "avg_latency_ms": 6230,
    ...
  },
  "threshold_check": {
    "recall@3": {"target": 0.85, "actual": 0.87, "pass": true},
    ...
  },
  "overall_pass": true,
  "failed_questions": ["q023", "q067"],
  "benchmark_version": "benchmark"
}
```

### HTML Report

Visual report showing:
- Summary cards (total, pass rate, status)
- Metrics table with pass/fail indicators
- Failed questions list
- Git commit information

## A/B Comparison Workflow

1. Run baseline evaluation before making changes:
   ```bash
   python evaluation/run_eval.py run --benchmark evaluation/benchmark.json --save-db
   python evaluation/run_eval.py set-baseline eval_20260421_001
   ```

2. Make code changes (e.g., improve retrieval, adjust chunking)

3. Run new evaluation:
   ```bash
   python evaluation/run_eval.py run --benchmark evaluation/benchmark.json --save-db
   ```

4. Compare results:
   ```bash
   python evaluation/run_eval.py compare eval_20260421_001 eval_20260421_002
   ```

5. Check comparison report to see if changes improved metrics

## Integration with Development Workflow

### Before Making Changes

Run baseline evaluation to establish current performance:

```bash
python evaluation/run_eval.py run --benchmark evaluation/benchmark.json --ragas --save-db
python evaluation/run_eval.py set-baseline <eval_id>
```

### After Making Changes

Run evaluation again and compare:

```bash
python evaluation/run_eval.py run --benchmark evaluation/benchmark.json --ragas --save-db
python evaluation/run_eval.py compare <baseline_id> <current_id>
```

### Decision Criteria

- If `overall_improvement = true`: Accept changes
- If key metrics degraded: Reject or investigate
- If mixed results: Manual judgment needed

## Dependencies

Added to requirements.txt:
- `ragas>=0.1.0` - LLM quality evaluation
- `pandas>=2.0.0` - Report analysis
- `jinja2>=3.1.0` - HTML template rendering
- `datasets>=2.14.0` - Ragas dependency

Install: `pip install -r requirements.txt`

## Troubleshooting

### Module Import Error

Set PYTHONPATH before running:

```bash
$env:PYTHONPATH="E:\newprojects\firstmodel"
python evaluation/run_eval.py run
```

### Ragas Not Available

Use mock evaluator for testing:

```bash
python evaluation/run_eval.py run --mock-ragas
```

### API Connection Failed

Check if service is running:

```bash
curl http://localhost:8000/health
```

Or specify custom endpoint:

```bash
python evaluation/run_eval.py run --api-endpoint http://your-server:8000/query
```

## Next Steps

- Phase 2: Full Ragas integration for faithfulness/relevancy
- Phase 3: Extend benchmark to 100+ questions with edge cases
- Phase 4: Flow and rejection metric improvements
- Phase 5: CI/CD integration and Feishu notification

## References

- Plan document: `.kilo/plans/1776750437113-nimble-panda.md`
- Notes: `notes.md` - System requirements
- Rateways: `rateways.md` - Metric specifications
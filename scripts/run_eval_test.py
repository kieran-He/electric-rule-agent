"""Run evaluation with server"""
import subprocess
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

server_proc = subprocess.Popen(
    ["python", "-m", "uvicorn", "app.main:app", "--host", "localhost", "--port", "8000"],
)

time.sleep(3)

try:
    from evaluation.run_eval import run_evaluation
    
    report = run_evaluation(
        benchmark_path="evaluation/benchmark_test.json",
        api_endpoint="http://localhost:8000/query",
        use_ragas=False,
        use_mock_ragas=True,
        output_dir="evaluation/reports",
    )
    
    print(f"\nEvaluation Report: {report.eval_id}")
    print(f"Total Questions: {report.total_questions}")
    print(f"Overall Pass: {report.overall_pass}")
    print("\nMetrics:")
    for metric, value in report.metrics.items():
        check = report.threshold_check.get(metric, {})
        status = "[OK]" if check.get("pass", True) else "[FAIL]"
        print(f"  {status} {metric}: {value:.4f}")
    
finally:
    server_proc.terminate()
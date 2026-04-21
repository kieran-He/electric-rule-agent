from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from evaluation.benchmark_generator import BenchmarkGenerator
from evaluation.evaluator import RAGEvaluator, EvaluationReport
from evaluation.ragas_evaluator import get_ragas_evaluator
from evaluation.report_generator import ReportGenerator


def get_api_endpoint() -> str:
    host = os.getenv("HOST", "localhost")
    port = os.getenv("PORT", "8000")
    return f"http://{host}:{port}/query"


def get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return None


def run_evaluation(
    benchmark_path: str,
    api_endpoint: Optional[str] = None,
    category: Optional[str] = None,
    use_ragas: bool = False,
    use_mock_ragas: bool = False,
    ragas_config_path: Optional[str] = None,
    save_to_db: bool = False,
    output_dir: str = "evaluation/reports",
) -> EvaluationReport:
    endpoint = api_endpoint or get_api_endpoint()
    
    ragas_evaluator = None
    ragas_config = None
    
    if use_ragas:
        # Load Ragas config
        from evaluation.ragas_config import RagasConfig
        
        if ragas_config_path:
            ragas_config = RagasConfig.from_file(ragas_config_path)
        else:
            ragas_config = RagasConfig.from_env()
        
        # Override use_mock if specified
        if use_mock_ragas:
            ragas_config.use_mock = True
        
        # Validate config
        if not ragas_config.validate():
            print("Warning: Invalid Ragas config, using mock evaluator")
            ragas_config.use_mock = True
        
        # Export config to environment
        ragas_config.to_env()
        
        # Get evaluator
        ragas_evaluator = get_ragas_evaluator(
            use_mock=ragas_config.use_mock,
            llm_endpoint=ragas_config.llm_endpoint,
            llm_api_key=ragas_config.llm_api_key,
        )
        
        print(f"Ragas config: enabled={ragas_config.enabled}, mock={ragas_config.use_mock}")
    
    evaluator = RAGEvaluator(
        api_endpoint=endpoint,
        ragas_evaluator=ragas_evaluator,
        ragas_config=ragas_config,
    )
    
    report = evaluator.run_benchmark(
        benchmark_path=benchmark_path,
        category_filter=category,
        use_ragas=use_ragas,
    )
    
    report.git_commit = get_git_commit()
    
    report_generator = ReportGenerator(output_dir=output_dir)
    report_generator.save_json_report(report)
    report_generator.save_html_report(report)
    
    if save_to_db:
        try:
            from app.db.session import SessionLocal
            evaluator.session_factory = SessionLocal
            evaluator.save_to_database(report)
            print(f"Saved to database: {report.eval_id}")
        except Exception as e:
            print(f"Failed to save to database: {e}")
    
    return report


def generate_benchmark(
    docs_path: str,
    output_path: str,
    total_count: int = 100,
) -> None:
    generator = BenchmarkGenerator()
    generator.generate_from_docs(
        docs_path=docs_path,
        output_path=output_path,
        total_count=total_count,
    )
    print(f"Generated benchmark: {output_path}")


def compare_evaluations(
    eval_id_1: str,
    eval_id_2: str,
    reports_dir: str = "evaluation/reports",
) -> dict:
    report_generator = ReportGenerator(output_dir=reports_dir)
    
    baseline_path = Path(reports_dir) / f"{eval_id_1}.json"
    current_path = Path(reports_dir) / f"{eval_id_2}.json"
    
    if not baseline_path.exists():
        print(f"Baseline report not found: {baseline_path}")
        return {}
    
    if not current_path.exists():
        print(f"Current report not found: {current_path}")
        return {}
    
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    
    with open(current_path, "r", encoding="utf-8") as f:
        current_data = json.load(f)
    
    baseline_report = EvaluationReport(
        eval_id=baseline_data["eval_id"],
        timestamp=baseline_data["timestamp"],
        total_questions=baseline_data["total_questions"],
        metrics=baseline_data["metrics"],
        threshold_check=baseline_data["threshold_check"],
        overall_pass=baseline_data["overall_pass"],
        failed_questions=baseline_data.get("failed_questions", []),
        benchmark_version=baseline_data.get("benchmark_version", "v1.0"),
    )
    
    current_report = EvaluationReport(
        eval_id=current_data["eval_id"],
        timestamp=current_data["timestamp"],
        total_questions=current_data["total_questions"],
        metrics=current_data["metrics"],
        threshold_check=current_data["threshold_check"],
        overall_pass=current_data["overall_pass"],
        failed_questions=current_data.get("failed_questions", []),
        benchmark_version=current_data.get("benchmark_version", "v1.0"),
    )
    
    evaluator = RAGEvaluator(api_endpoint="")
    comparison = evaluator.compare_reports(baseline_report, current_report)
    
    comparison_path = Path(reports_dir) / f"comparison_{eval_id_1}_{eval_id_2}.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print(f"Comparison saved: {comparison_path}")
    
    return comparison


def set_baseline(
    eval_id: str,
    reports_dir: str = "evaluation/reports",
) -> None:
    baseline_file = Path(reports_dir) / "baseline.json"
    
    report_path = Path(reports_dir) / f"{eval_id}.json"
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return
    
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    data["is_baseline"] = True
    data["baseline_set_at"] = datetime.now().isoformat()
    
    with open(baseline_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Baseline set: {eval_id}")


def main():
    parser = argparse.ArgumentParser(
        description="RAG System Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    eval_parser = subparsers.add_parser("run", help="Run evaluation")
    eval_parser.add_argument(
        "--benchmark",
        type=str,
        default="evaluation/benchmark.json",
        help="Path to benchmark JSON file",
    )
    eval_parser.add_argument(
        "--api-endpoint",
        type=str,
        help="Query API endpoint URL",
    )
    eval_parser.add_argument(
        "--category",
        type=str,
        help="Filter by category (clause_qa, flow_qa, compare_qa, settlement_qa, rejection)",
    )
    eval_parser.add_argument(
        "--ragas",
        action="store_true",
        help="Use Ragas for LLM quality evaluation",
    )
    eval_parser.add_argument(
        "--mock-ragas",
        action="store_true",
        help="Use mock Ragas evaluator (for testing)",
    )
    eval_parser.add_argument(
        "--ragas-config",
        type=str,
        default="evaluation/ragas_config.json",
        help="Path to Ragas configuration file",
    )
    eval_parser.add_argument(
        "--save-db",
        action="store_true",
        help="Save results to database",
    )
    eval_parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/reports",
        help="Output directory for reports",
    )
    
    gen_parser = subparsers.add_parser("generate", help="Generate benchmark")
    gen_parser.add_argument(
        "--docs-path",
        type=str,
        default="data/raw",
        help="Path to knowledge base documents",
    )
    gen_parser.add_argument(
        "--output",
        type=str,
        default="evaluation/benchmark.json",
        help="Output path for benchmark JSON",
    )
    gen_parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Total number of questions to generate",
    )
    
    compare_parser = subparsers.add_parser("compare", help="Compare two evaluations")
    compare_parser.add_argument(
        "eval_id_1",
        type=str,
        help="Baseline evaluation ID",
    )
    compare_parser.add_argument(
        "eval_id_2",
        type=str,
        help="Current evaluation ID",
    )
    compare_parser.add_argument(
        "--reports-dir",
        type=str,
        default="evaluation/reports",
        help="Directory containing reports",
    )
    
    baseline_parser = subparsers.add_parser("set-baseline", help="Set baseline evaluation")
    baseline_parser.add_argument(
        "eval_id",
        type=str,
        help="Evaluation ID to set as baseline",
    )
    baseline_parser.add_argument(
        "--reports-dir",
        type=str,
        default="evaluation/reports",
        help="Directory containing reports",
    )
    
    args = parser.parse_args()
    
    if args.command == "run":
        report = run_evaluation(
            benchmark_path=args.benchmark,
            api_endpoint=args.api_endpoint,
            category=args.category,
            use_ragas=args.ragas,
            use_mock_ragas=args.mock_ragas,
            ragas_config_path=args.ragas_config if args.ragas else None,
            save_to_db=args.save_db,
            output_dir=args.output_dir,
        )
        
        print(f"\nEvaluation Report: {report.eval_id}")
        print(f"Total Questions: {report.total_questions}")
        print(f"Overall Pass: {report.overall_pass}")
        print("\nMetrics:")
        for metric, value in report.metrics.items():
            check = report.threshold_check.get(metric, {})
            status = "[OK]" if check.get("pass", True) else "[FAIL]"
            print(f"  {status} {metric}: {value:.4f} (target: {check.get('target', 'N/A')})")
        
        if report.failed_questions:
            print(f"\nFailed Questions: {len(report.failed_questions)}")
            print(f"  IDs: {', '.join(report.failed_questions[:10])}...")
    
    elif args.command == "generate":
        generate_benchmark(
            docs_path=args.docs_path,
            output_path=args.output,
            total_count=args.count,
        )
    
    elif args.command == "compare":
        comparison = compare_evaluations(
            eval_id_1=args.eval_id_1,
            eval_id_2=args.eval_id_2,
            reports_dir=args.reports_dir,
        )
        
        print(f"\nComparison Report")
        print(f"Baseline: {comparison.get('baseline_eval_id', 'N/A')}")
        print(f"Current: {comparison.get('current_eval_id', 'N/A')}")
        print(f"Overall Improvement: {comparison.get('overall_improvement', False)}")
        print(f"Recommendation: {comparison.get('recommendation', 'N/A')}")
        
        print("\nMetric Changes:")
        for metric, change in comparison.get("metric_changes", {}).items():
            delta = change.get("delta", 0)
            improved = change.get("improved", False)
            sign = "+" if delta > 0 else ""
            status = "[UP]" if improved else "[DOWN]"
            print(f"  {status} {metric}: {sign}{delta:.4f}")
    
    elif args.command == "set-baseline":
        set_baseline(
            eval_id=args.eval_id,
            reports_dir=args.reports_dir,
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
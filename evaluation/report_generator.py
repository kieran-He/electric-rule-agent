from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA_AVAILABLE = True
except ImportError:
    JINJA_AVAILABLE = False
    Environment = None
    FileSystemLoader = None
    Template = None


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Report - {{ eval_id }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        .header .meta {
            font-size: 14px;
            opacity: 0.9;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .summary-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-card h3 {
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
        }
        .summary-card .value {
            font-size: 24px;
            font-weight: bold;
        }
        .pass { color: #27ae60; }
        .fail { color: #e74c3c; }
        .metrics-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .metrics-section h2 {
            font-size: 20px;
            margin-bottom: 15px;
            color: #333;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
        }
        .metrics-table th,
        .metrics-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .metrics-table th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .metrics-table tr:hover {
            background-color: #f5f5f5;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-pass {
            background-color: #d4edda;
            color: #155724;
        }
        .status-fail {
            background-color: #f8d7da;
            color: #721c24;
        }
        .failed-questions {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .failed-questions h2 {
            font-size: 20px;
            margin-bottom: 15px;
            color: #333;
        }
        .question-list {
            list-style: none;
        }
        .question-list li {
            padding: 10px;
            margin-bottom: 8px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .question-list .id {
            font-weight: bold;
            color: #667eea;
        }
        .question-list .text {
            color: #666;
            margin-top: 4px;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RAG System Evaluation Report</h1>
            <div class="meta">
                <span>Evaluation ID: {{ eval_id }}</span> |
                <span>Timestamp: {{ timestamp }}</span> |
                <span>Benchmark Version: {{ benchmark_version }}</span>
            </div>
        </div>

        <div class="summary">
            <div class="summary-card">
                <h3>Total Questions</h3>
                <div class="value">{{ total_questions }}</div>
            </div>
            <div class="summary-card">
                <h3>Pass Count</h3>
                <div class="value">{{ pass_count }}</div>
            </div>
            <div class="summary-card">
                <h3>Pass Rate</h3>
                <div class="value">{{ "%.2f%%" % (pass_rate * 100) }}</div>
            </div>
            <div class="summary-card">
                <h3>Overall Status</h3>
                <div class="value {% if overall_pass %}pass{% else %}fail{% endif %}">
                    {% if overall_pass %}PASS{% else %}FAIL{% endif %}
                </div>
            </div>
        </div>

        <div class="metrics-section">
            <h2>Metrics Summary</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Actual Value</th>
                        <th>Target</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metric, check in threshold_check.items() %}
                    <tr>
                        <td>{{ metric }}</td>
                        <td>{{ "%.4f" % check.actual if check.actual != None else "N/A" }}</td>
                        <td>{{ check.target }}</td>
                        <td>
                            <span class="status-badge {% if check.pass %}status-pass{% else %}status-fail{% endif %}">
                                {% if check.pass %}PASS{% else %}FAIL{% endif %}
                            </span>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        {% if ragas_metrics %}
        <div class="metrics-section">
            <h2>LLM Quality Metrics (Ragas)</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Average Score</th>
                        <th>Target</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metric, value in ragas_metrics.items() %}
                    <tr>
                        <td>{{ metric }}</td>
                        <td>{{ "%.2f" % value }}</td>
                        <td>≥0.85</td>
                        <td>
                            <span class="status-badge {% if value >= 0.85 %}status-pass{% else %}status-fail{% endif %}">
                                {% if value >= 0.85 %}PASS{% else %}FAIL{% endif %}
                            </span>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            <p style="color: #666; margin-top: 10px;">
                Ragas evaluates LLM output quality: faithfulness (answer based on context), 
                answer_relevancy (answer addresses question), context_precision (context relevance).
            </p>
        </div>
        {% endif %}

        {% if failed_questions %}
        <div class="failed-questions">
            <h2>Failed Questions ({{ failed_questions|length }})</h2>
            <ul class="question-list">
                {% for question_id in failed_questions[:20] %}
                <li>
                    <div class="id">{{ question_id }}</div>
                </li>
                {% endfor %}
            </ul>
            {% if failed_questions|length > 20 %}
            <p style="color: #666; margin-top: 10px;">
                ... and {{ failed_questions|length - 20 }} more failed questions
            </p>
            {% endif %}
        </div>
        {% endif %}

        <div class="footer">
            <p>Generated by RAG Evaluation System</p>
            <p>Git Commit: {{ git_commit or "N/A" }}</p>
        </div>
    </div>
</body>
</html>
"""


COMPARISON_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A/B Comparison Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        .header .meta {
            font-size: 14px;
            opacity: 0.9;
        }
        .recommendation {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            text-align: center;
        }
        .recommendation h2 {
            font-size: 24px;
            margin-bottom: 10px;
        }
        .improved { color: #27ae60; }
        .degraded { color: #e74c3c; }
        .neutral { color: #f39c12; }
        .metrics-table {
            width: 100%;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }
        .metrics-table table {
            width: 100%;
            border-collapse: collapse;
        }
        .metrics-table th,
        .metrics-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .metrics-table th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .delta-positive { color: #27ae60; }
        .delta-negative { color: #e74c3c; }
        .delta-neutral { color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>A/B Comparison Report</h1>
            <div class="meta">
                <span>Baseline: {{ baseline_eval_id }}</span> |
                <span>Current: {{ current_eval_id }}</span>
            </div>
        </div>

        <div class="recommendation">
            <h2 class="{% if overall_improvement %}improved{% else %}degraded{% endif %}">
                {% if overall_improvement %}IMPROVED{% else %}DEGRADED{% endif %}
            </h2>
            <p>{{ recommendation }}</p>
        </div>

        <div class="metrics-table">
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Baseline</th>
                        <th>Current</th>
                        <th>Delta</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metric, change in metric_changes.items() %}
                    <tr>
                        <td>{{ metric }}</td>
                        <td>{{ "%.4f" % change.baseline }}</td>
                        <td>{{ "%.4f" % change.current }}</td>
                        <td class="{% if change.delta > 0 %}delta-positive{% elif change.delta < 0 %}delta-negative{% else %}delta-neutral{% endif %}">
                            {{ "+" if change.delta > 0 else "" }}{{ "%.4f" % change.delta }}
                        </td>
                        <td>
                            {% if change.improved %}↑{% else %}↓{% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""


class ReportGenerator:
    def __init__(self, output_dir: str = "evaluation/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json_report(self, report: Any) -> Path:
        report_data = {
            "eval_id": report.eval_id,
            "timestamp": report.timestamp,
            "total_questions": report.total_questions,
            "metrics": report.metrics,
            "threshold_check": report.threshold_check,
            "overall_pass": report.overall_pass,
            "failed_questions": report.failed_questions,
            "benchmark_version": report.benchmark_version,
            "git_commit": report.git_commit,
            "config_snapshot": report.config_snapshot,
        }
        
        if hasattr(report, "comparison_with_baseline") and report.comparison_with_baseline:
            report_data["comparison_with_baseline"] = report.comparison_with_baseline
        
        output_path = self.output_dir / f"{report.eval_id}.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        return output_path

    def save_html_report(self, report: Any) -> Path:
        pass_count = report.pass_count if hasattr(report, "pass_count") else len([r for r in report.results if r.is_correct])
        pass_rate = pass_count / report.total_questions if report.total_questions > 0 else 0
        
        # Extract Ragas metrics if available
        ragas_metrics = {}
        if hasattr(report, "metrics"):
            for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
                if metric in report.metrics:
                    ragas_metrics[metric] = report.metrics[metric]
        
        template_context = {
            "eval_id": report.eval_id,
            "timestamp": report.timestamp,
            "total_questions": report.total_questions,
            "pass_count": pass_count,
            "pass_rate": pass_rate,
            "overall_pass": report.overall_pass,
            "threshold_check": report.threshold_check,
            "failed_questions": report.failed_questions,
            "benchmark_version": report.benchmark_version,
            "git_commit": report.git_commit,
            "ragas_metrics": ragas_metrics if ragas_metrics else None,
        }
        
        if JINJA_AVAILABLE:
            env = Environment()
            template = env.from_string(HTML_TEMPLATE)
            html_content = template.render(**template_context)
        else:
            html_content = self._render_html_simple(template_context)
        
        output_path = self.output_dir / f"{report.eval_id}.html"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return output_path

    def _render_html_simple(self, context: Dict[str, Any]) -> str:
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>RAG Evaluation Report - {context['eval_id']}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #667eea; color: white; padding: 20px; }
        .metrics { margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; }
        th { background: #f8f9fa; }
        .pass { color: green; }
        .fail { color: red; }
    </style>
</head>
<body>
    <div class="header">
        <h1>RAG Evaluation Report</h1>
        <p>Evaluation ID: {context['eval_id']} | Timestamp: {context['timestamp']}</p>
    </div>
    
    <div class="metrics">
        <h2>Summary</h2>
        <p>Total Questions: {context['total_questions']}</p>
        <p>Overall Pass: <span class="{('pass' if context['overall_pass'] else 'fail')}">{('PASS' if context['overall_pass'] else 'FAIL')}</span></p>
        
        <h2>Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Actual</th><th>Target</th><th>Status</th></tr>
"""
        
        for metric, check in context["threshold_check"].items():
            actual = f"{check['actual']:.4f}" if check['actual'] is not None else "N/A"
            status_class = "pass" if check['pass'] else "fail"
            status_text = "PASS" if check['pass'] else "FAIL"
            html += f"<tr><td>{metric}</td><td>{actual}</td><td>{check['target']}</td><td class='{status_class}'>{status_text}</td></tr>"
        
        html += """
        </table>
    </div>
</body>
</html>
"""
        return html

    def save_comparison_report(
        self,
        comparison: Dict[str, Any],
        baseline_eval_id: str,
        current_eval_id: str,
    ) -> Path:
        json_path = self.output_dir / f"comparison_{baseline_eval_id}_{current_eval_id}.json"
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        
        html_path = self.output_dir / f"comparison_{baseline_eval_id}_{current_eval_id}.html"
        
        if JINJA_AVAILABLE:
            env = Environment()
            template = env.from_string(COMPARISON_TEMPLATE)
            html_content = template.render(**comparison)
        else:
            html_content = self._render_comparison_html_simple(comparison)
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return html_path

    def _render_comparison_html_simple(self, comparison: Dict[str, Any]) -> str:
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>A/B Comparison Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #667eea; color: white; padding: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; }
        th { background: #f8f9fa; }
        .improved { color: green; }
        .degraded { color: red; }
    </style>
</head>
<body>
    <div class="header">
        <h1>A/B Comparison Report</h1>
        <p>Baseline: {comparison.get('baseline_eval_id', 'N/A')} | Current: {comparison.get('current_eval_id', 'N/A')}</p>
    </div>
    
    <h2 class="{('improved' if comparison.get('overall_improvement') else 'degraded')}">
        {('IMPROVED' if comparison.get('overall_improvement') else 'DEGRADED')}
    </h2>
    <p>{comparison.get('recommendation', 'N/A')}</p>
    
    <table>
        <tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Delta</th></tr>
"""
        
        for metric, change in comparison.get("metric_changes", {}).items():
            baseline = f"{change['baseline']:.4f}"
            current = f"{change['current']:.4f}"
            delta = f"+{change['delta']:.4f}" if change['delta'] > 0 else f"{change['delta']:.4f}"
            delta_class = "improved" if change['improved'] else "degraded"
            html += f"<tr><td>{metric}</td><td>{baseline}</td><td>{current}</td><td class='{delta_class}'>{delta}</td></tr>"
        
        html += """
    </table>
</body>
</html>
"""
        return html

    def load_report(self, eval_id: str) -> Optional[Dict[str, Any]]:
        report_path = self.output_dir / f"{eval_id}.json"
        
        if not report_path.exists():
            return None
        
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_reports(self) -> List[str]:
        return [
            f.stem for f in self.output_dir.glob("*.json")
            if not f.stem.startswith("comparison") and not f.stem == "baseline"
        ]

    def get_baseline(self) -> Optional[Dict[str, Any]]:
        baseline_path = self.output_dir / "baseline.json"
        
        if not baseline_path.exists():
            return None
        
        with open(baseline_path, "r", encoding="utf-8") as f:
            return json.load(f)
#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import json

report_path = Path('evaluation/reports_minimax/eval_20260421_173535_0f25a465.json')
if not report_path.exists():
    # Find latest report
    reports_dir = Path('evaluation/reports_minimax')
    reports = sorted(reports_dir.glob('*.json'))
    if reports:
        report_path = reports[-1]
        print(f'Using latest report: {report_path.name}')
    else:
        print('No reports found')
        sys.exit(1)

with open(report_path, encoding='utf-8') as f:
    data = json.load(f)

print('Evaluation Summary:')
print('='*60)
print(f'Eval ID: {data.get("eval_id", "")}')
print(f'Total Questions: {data.get("total_questions", 0)}')
print(f'Overall Pass: {data.get("overall_pass", False)}')

print('\nMetrics:')
for k, v in data.get('metrics', {}).items():
    check = data.get('threshold_check', {}).get(k, {})
    target = check.get('target', 'N/A')
    passed = check.get('pass', True)
    status = '[OK]' if passed else '[FAIL]'
    if v is not None:
        print(f'  {status} {k}: {v:.4f} (target: {target})')
    else:
        print(f'  {status} {k}: N/A (target: {target})')

print('\nFailed questions:')
for qid in data.get('failed_questions', []):
    print(f'  - {qid}')

# Check if there are results details
if 'results' in data:
    print('\nResults details:')
    for result in data.get('results', [])[:2]:
        print(f'  Q: {result.get("question", "")[:50]}...')
        print(f'  Expected docs: {result.get("expected_docs", [])[:1]}')
        print(f'  Predicted docs: {result.get("predicted_docs", [])[:1]}')
        print(f'  Answer: {result.get("answer", "")[:100]}...')
        print()
#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import json

report_path = Path('evaluation/reports_final/eval_20260421_170217_5c4f36de.json')
with open(report_path, encoding='utf-8') as f:
    data = json.load(f)

print('Evaluation Summary:')
print('='*60)
print(f'Eval ID: {data.get("eval_id", "")}')
print(f'Total Questions: {data.get("total_questions", 0)}')
print(f'Overall Pass: {data.get("overall_pass", False)}')

print('\\nMetrics:')
for k, v in data.get('metrics', {}).items():
    check = data.get('threshold_check', {}).get(k, {})
    target = check.get('target', 'N/A')
    passed = check.get('pass', True)
    status = '[OK]' if passed else '[FAIL]'
    print(f'  {status} {k}: {v:.4f} (target: {target})')

print('\\nFailed questions:')
for qid in data.get('failed_questions', []):
    print(f'  - {qid}')
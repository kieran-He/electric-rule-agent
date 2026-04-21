#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json

reports_dir = Path('evaluation/reports_minimax2')
reports = sorted(reports_dir.glob('*.json'))
if not reports:
    print('No reports found')
    sys.exit(1)

report_path = reports[-1]
print(f'Report: {report_path.name}\n')

with open(report_path, encoding='utf-8') as f:
    data = json.load(f)

print('Metrics Summary:')
for k, v in data.get('metrics', {}).items():
    if v is not None:
        print(f'  {k}: {v:.2f}')

# Show rejection question result
print('\nRejection Question (q005):')
for result in data.get('results', []):
    if result.get('question_id') == 'q005':
        print(f'  Question: {result.get("question", "")}')
        print(f'  Should reject: {result.get("should_reject", False)}')
        print(f'  Answer: {result.get("answer", "")[:200]}...')
        print(f'  Keywords hit: {result.get("keywords_hit", False)}')
        print(f'  Expected keywords: {result.get("expected_keywords", [])}')
        break

print('\nLatency Analysis:')
latencies = [r.get('latency_ms', 0) for r in data.get('results', [])]
print(f'  Min: {min(latencies)}ms')
print(f'  Max: {max(latencies)}ms')
print(f'  Avg: {sum(latencies)/len(latencies):.0f}ms')
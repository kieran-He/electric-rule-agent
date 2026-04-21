#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

content = Path('evaluation/metrics.py').read_text(encoding='utf-8')
print('Searching for rejection metrics...')
for i, line in enumerate(content.split('\n'), 1):
    if 'rejection' in line.lower():
        print(f'{i}: {line}')
#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import json
from pathlib import Path

processed_dir = Path('data/processed')
for f in processed_dir.glob('*.json'):
    if not f.name.startswith('_'):
        with open(f, encoding='utf-8') as fp:
            data = json.load(fp)
        doc_name = data.get('metadata', {}).get('doc_name', '')
        if '2026年' in doc_name and '交易实施方案' in doc_name:
            print(f'File: {f.name}')
            print(f'Doc: {doc_name}')
            print(f'Clauses: {len(data.get("clauses", []))}')
            
            # Search for clauses about 签约比例
            found = 0
            for clause in data.get('clauses', []):
                text = clause.get('clause_text', '')
                if '签约比例' in text or ('签约' in text and '比例' in text):
                    found += 1
                    print()
                    print(f'  [{found}] Title: {clause.get("title_path", "")[:60]}')
                    print(f'      Article: {clause.get("article_no", "")}')
                    print(f'      Text: {text[:300]}...')
            
            if found == 0:
                print('\\n  No clauses found with 签约比例')
            break
#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import requests
import json
import time

start = time.time()
try:
    resp = requests.post(
        'http://localhost:8000/query',
        json={
            'query': '陕西2026年中长期签约比例',
            'session_id': 'test_001',
            'province_codes': ['SN'],
            'mode': 'province_only',
            'top_k': 5,
            'need_citation': True
        },
        headers={'Content-Type': 'application/json'},
        timeout=120,
    )
    elapsed = time.time() - start
    print(f'Status: {resp.status_code}')
    print(f'Time: {elapsed:.2f}s')
    
    if resp.status_code == 200:
        data = resp.json()
        print(f'Answer: {data.get("answer", "")[:200]}...')
        print(f'Citations: {len(data.get("citations", []))}')
        print(f'Trace ID: {data.get("trace_id", "")}')
        print(f'Used Documents: {data.get("used_documents", [])[:2]}')
    else:
        print(f'Error response: {resp.text[:500]}')
except requests.Timeout:
    print(f'Timeout after {time.time() - start:.2f}s')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')
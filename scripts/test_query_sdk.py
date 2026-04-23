#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests
import time

start = time.time()
try:
    resp = requests.post(
        'http://localhost:8000/query',
        json={
            'query': '发电侧中长期合同签约比例要求是什么？',
            'session_id': 'test_sdk',
            'province_codes': ['SN'],
            'mode': 'province_only',
            'top_k': 3,
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
        answer = data.get("answer", "")
        print(f'Answer: {answer[:200]}...')
        print(f'Citations: {len(data.get("citations", []))}')
    else:
        print(f'Error: {resp.text[:200]}')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')
    print(f'Time: {time.time() - start:.2f}s')
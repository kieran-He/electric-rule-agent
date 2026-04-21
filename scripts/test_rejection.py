#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests
import time

# Test rejection question
start = time.time()
try:
    resp = requests.post(
        'http://localhost:8000/query',
        json={
            'query': '美国电力市场的监管政策是什么？',
            'session_id': 'test_reject',
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
        print(f'Answer: {answer[:300]}...')
        print(f'Citations: {len(data.get("citations", []))}')
        # Check if it's a proper rejection
        if '未检索到' in answer or '无法回答' in answer or '不在知识库' in answer:
            print('[REJECTION DETECTED - GOOD!]')
        else:
            print('[NO REJECTION - SHOULD REJECT THIS QUESTION]')
    else:
        print(f'Error: {resp.text[:200]}')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')
    print(f'Time: {time.time() - start:.2f}s')
#!/usr/bin/env python3
"""
Check MiniMax API response structure for thinking blocks
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import requests
import os
import time

api_key = os.getenv('LLM_API_KEY', '')
endpoint = os.getenv('LLM_ENDPOINT', 'https://api.minimaxi.com/anthropic/v1/messages')
model = os.getenv('LLM_MODEL', 'minimax-2.7')

print(f'API Key: {api_key[:20]}...')
print(f'Endpoint: {endpoint}')
print(f'Model: {model}')

headers = {
    'x-api-key': api_key,
    'anthropic-version': '2023-06-01',
    'Content-Type': 'application/json'
}

payload = {
    'model': model,
    'max_tokens': 512,
    'system': '你是电力政策问答助手。',
    'messages': [{'role': 'user', 'content': '你好'}]
}

print('\nTesting MiniMax API response structure...')
start = time.time()
session = requests.Session()
session.trust_env = False
resp = session.post(endpoint, json=payload, headers=headers, timeout=30)
elapsed = time.time() - start

print(f'Status: {resp.status_code}')
print(f'Time: {elapsed:.2f}s')

if resp.status_code == 200:
    data = resp.json()
    print(f'\nResponse keys: {list(data.keys())}')
    
    content_blocks = data.get('content', [])
    print(f'Content blocks count: {len(content_blocks)}')
    
    for i, block in enumerate(content_blocks):
        block_type = block.get('type', 'unknown')
        print(f'\n  Block {i}: type={block_type}')
        
        if block_type == 'text':
            text = block.get('text', '')
            print(f'    Text length: {len(text)}')
            print(f'    Text preview: {text[:100]}...')
        elif block_type == 'thinking':
            thinking_text = block.get('thinking', '')
            print(f'    THINKING FOUND!')
            print(f'    Thinking length: {len(thinking_text)}')
            print(f'    Thinking preview: {thinking_text[:100]}...')
    
    # Check if there's a thinking parameter we can disable
    print('\n\nChecking available parameters:')
    print('Current payload keys:', list(payload.keys()))
    
    # Test with thinking disabled (if such parameter exists)
    test_payloads = [
        {'thinking': False},
        {'extended_thinking': False},
        {'enable_thinking': False},
    ]
    
    for test_param in test_payloads:
        test_payload = payload.copy()
        test_payload.update(test_param)
        print(f'\n  Testing {test_param}...')
        try:
            resp2 = session.post(endpoint, json=test_payload, headers=headers, timeout=10)
            if resp2.status_code == 200:
                data2 = resp2.json()
                blocks = data2.get('content', [])
                has_thinking = any(b.get('type') == 'thinking' for b in blocks)
                print(f'    Status: 200, Thinking blocks: {has_thinking}')
            else:
                print(f'    Status: {resp2.status_code}')
        except Exception as e:
            print(f'    Error: {e}')
else:
    print(f'Error: {resp.text[:300]}')
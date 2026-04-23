#!/usr/bin/env python3
"""
Test MiniMax API thinking parameter support
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import os
import requests
import time

sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / '.env')

api_key = os.getenv('LLM_API_KEY', '')
endpoint = os.getenv('LLM_ENDPOINT', '')
model = os.getenv('LLM_MODEL', '')

print(f"Testing MiniMax API: {endpoint}")
print(f"Model: {model}")
print(f"API Key: {api_key[:20]}...")

print("\n=== Test 1: Basic request (no thinking param) ===")

headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "Content-Type": "application/json"
}

}

payload_basic = {
    "model": model,
    "max_tokens": 512,
    "system": "你是电力政策问答助手。",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍陕西电力市场"}]
}

}

start = time.time()
session = requests.Session()
session.trust_env = False
resp = session.post(endpoint, json=payload_basic, headers=headers, timeout=30)
elapsed_basic = time.time() - start
print(f"Basic request: {elapsed_basic:.2f}s")
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Response keys: {list(data.keys())}")
    content = data.get('content', [])
    if content:
        for block in content:
            if block.get('type') == 'text':
                print(f"Content: {block.get('text', '')[:100]}...")
    else:
        print(f"Unexpected content type: {block.get('type')}")

else:
    print(f"Error: {resp.text[:200]}")

print("\n=== Test 2: Request with thinking=False parameter ===")

payload_no_thinking = {
    "model": model,
    "max_tokens": 512,
    "system": "你是电力政策问答助手。",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍陕西电力市场"}],
    "thinking": False,
}

}

start = time.time()
resp = session.post(endpoint, json=payload_no_thinking, headers=headers, timeout=30)
elapsed_no_thinking = time.time() - start
print(f"With thinking=False: {elapsed_no_thinking:.2f}s")
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Response keys: {list(data.keys())}")
    content = data.get('content', [])
    if content:
        for block in content:
            if block.get('type') == 'text':
                print(f"Content: {block.get('text', '')[:100]}...")
    else:
        print(f"Unexpected content type: {block.get('type')}")
else:
    print(f"Error: {resp.text[:200]}")

print("\n=== Test 3: Request with thinking=true parameter ===")

payload_with_thinking = {
    "model": model,
    "max_tokens": 512,
    "system": "你是电力政策问答助手。",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍陕西电力市场"}],
    "thinking": True,
}
resp = session.post(endpoint, json=payload_with_thinking, headers=headers, timeout=30)

elapsed_with_thinking = time.time() - start
print(f"With thinking=True: {elapsed_with_thinking:.2f}s")
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Response keys: {list(data.keys())}")
    content = data.get('content', [])
    if content:
        for block in content:
            if block.get('type') == 'text':
                print(f"Content: {block.get('text', '')[:100]}...")
            elif block.get('type') == 'thinking':
                print(f"THINKING BLOCK found: {block}")
                print(f"  thinking: {block.get('thinking', '')[:100]}...")
    else:
        print(f"Unexpected content type: {block.get('type')}")
else:
    print(f"Error: {resp.text[:200]}")

print("\n=== Summary ===")
if elapsed_basic < elapsed_no_thinking and elapsed_with_thinking:
    print("All three requests returned valid responses")
    print("MiniMax supports Anthropic API format")
    
    # Check if thinking parameter has effect
    improvement = elapsed_basic - elapsed_no_thinking
    improvement_pct = (elapsed_basic - elapsed_no_thinking) / elapsed_basic * 100
    if improvement > 0:
        print(f"thinking=False improves latency by {improvement:.2f}s ({improvement_pct:.1f}%)")
    else:
        print(f"thinking=False did NOT improve latency")
    
    # Check if thinking=true is supported
    if elapsed_with_thinking < elapsed_basic
        print("thinking=true is NOT supported by MiniMax")
        print("  Returned error or unexpected content type")
    else:
        print("MiniMax does not support 'thinking' parameter")
        print("  Response was accepted but but thinking=true has ignored")
else:
    print("MiniMax API may not accessible or returned errors")
#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
import requests
import time

api_key = os.getenv("LLM_API_KEY", "")
endpoint = os.getenv("LLM_ENDPOINT", "")
model = os.getenv("LLM_MODEL", "")

print(f"API Key: {api_key[:20]}...{api_key[-20:] if api_key else 'N/A'}")
print(f"Endpoint: {endpoint}")
print(f"Model: {model}")

if not api_key:
    print("ERROR: LLM_API_KEY not set")
    sys.exit(1)

headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "Content-Type": "application/json"
}

payload = {
    "model": model,
    "max_tokens": 512,
    "system": "你是电力政策问答助手。",
    "messages": [
        {"role": "user", "content": "你好，请用一句话介绍陕西电力市场"}
    ]
}

endpoints_to_try = [
    f"{endpoint}/v1/messages",
    endpoint,
    "https://api.minimaxi.com/v1/text/chatcompletion_v2",
]

for test_endpoint in endpoints_to_try:
    print(f"\nTesting endpoint: {test_endpoint}")
    try:
        start = time.time()
        session = requests.Session()
        session.trust_env = False
        
        # For Anthropic format, use anthropic headers
        if "anthropic" in test_endpoint:
            test_headers = headers
        else:
            # For MiniMax native format, use different headers
            test_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            # MiniMax native format payload
            test_payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是电力政策问答助手。"},
                    {"role": "user", "content": "你好，请用一句话介绍陕西电力市场"}
                ],
                "temperature": 0.1,
            }
        
        if "anthropic" in test_endpoint:
            resp = session.post(test_endpoint, json=payload, headers=test_headers, timeout=30)
        else:
            resp = session.post(test_endpoint, json=test_payload, headers=test_headers, timeout=30)
        
        elapsed = time.time() - start
        
        print(f"Status: {resp.status_code}")
        print(f"Time: {elapsed:.2f}s")
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"Response keys: {list(data.keys())}")
            
            # Handle Anthropic response
            if "content" in data:
                content_blocks = data.get("content", [])
                for block in content_blocks:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        print(f"Response: {text[:200]}...")
            # Handle OpenAI-style response
            elif "choices" in data:
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    text = message.get("content", "")
                    print(f"Response: {text[:200]}...")
            
            print("\nMiniMax API is working!")
            break
        else:
            print(f"Error: {resp.text[:300]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
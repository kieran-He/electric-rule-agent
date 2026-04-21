#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import requests

api_key = os.getenv("GLM_API_KEY", "")
endpoint = os.getenv("GLM_ENDPOINT", "")
model = os.getenv("GLM_MODEL", "")

print(f"API Key: {api_key[:10]}...{api_key[-10:] if api_key else 'N/A'}")
print(f"Endpoint: {endpoint}")
print(f"Model: {model}")

if not api_key:
    print("ERROR: GLM_API_KEY not set")
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": model,
    "messages": [
        {"role": "user", "content": "你好，请用一句话介绍陕西电力市场"}
    ],
    "temperature": 0.1
}

try:
    print("\nTesting GLM API...")
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"Response: {content[:100]}...")
        print("\nGLM API is working!")
    else:
        print(f"Error: {resp.text[:200]}")
except Exception as e:
    print(f"Error: {e}")
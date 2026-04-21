#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
import requests

api_key = os.getenv("LLM_API_KEY", "")
print(f"API Key: {api_key[:20]}...")

# Try Anthropic-compatible endpoint
anthropic_endpoint = "https://api.minimaxi.com/anthropic/v1/messages"

headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "Content-Type": "application/json"
}

models_to_try = ["minimax-2.7", "minimax2.7", "claude-3-opus-20240229", "claude-3-sonnet-20240229"]

print("\nTesting Anthropic-compatible endpoint:")
for model in models_to_try:
    payload = {
        "model": model,
        "max_tokens": 512,
        "system": "你是电力政策问答助手。",
        "messages": [
            {"role": "user", "content": "你好，请用一句话介绍陕西电力市场"}
        ]
    }
    
    session = requests.Session()
    session.trust_env = False
    try:
        resp = session.post(anthropic_endpoint, json=payload, headers=headers, timeout=30)
        data = resp.json()
        
        if resp.status_code == 200:
            content_blocks = data.get("content", [])
            if content_blocks:
                text = ""
                for block in content_blocks:
                    if block.get("type") == "text":
                        text += block.get("text", "")
                print(f"  {model}: SUCCESS! Response: {text[:100]}...")
                break
            else:
                print(f"  {model}: status={resp.status_code}, no content blocks")
        else:
            error = data.get("error", {})
            error_type = error.get("type", "")
            error_msg = error.get("message", resp.text[:100])
            print(f"  {model}: status={resp.status_code}, error={error_type}, msg={error_msg[:80]}")
    except Exception as e:
        print(f"  {model}: Error - {type(e).__name__}: {str(e)[:80]}")
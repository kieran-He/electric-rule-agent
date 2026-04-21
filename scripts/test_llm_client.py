#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
from app.generator import LLMClient

api_key = os.getenv("LLM_API_KEY", "")
endpoint = os.getenv("LLM_ENDPOINT", "")
model = os.getenv("LLM_MODEL", "")
provider = os.getenv("LLM_PROVIDER", "openai")

print(f"API Key: {api_key[:20]}...")
print(f"Endpoint: {endpoint}")
print(f"Model: {model}")
print(f"Provider: {provider}")

client = LLMClient(
    api_key=api_key,
    endpoint=endpoint,
    model=model,
    provider=provider,
)

print(f"\nClient ready: {client.ready}")
print(f"Client mode: {client.mode}")

if client.ready:
    print("\nTesting LLMClient.generate_answer...")
    
    # First, let's see the raw response
    import requests
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    models_to_try = ["minimax-2.7", "minimax2.7", "MiniMax2.7", "minimax-27", "minimax27", "abab6.5s-chat"]
    
    print("Testing different models:")
    for test_model in models_to_try:
        payload = {
            "model": test_model,
            "messages": [
                {"role": "system", "content": "你是电力政策问答助手。"},
                {"role": "user", "content": "你好"}
            ],
            "temperature": 0.1,
        }
        
        session = requests.Session()
        session.trust_env = False
        try:
            resp = session.post(endpoint, json=payload, headers=headers, timeout=30)
            data = resp.json()
            choices = data.get("choices")
            status_msg = data.get("base_resp", {}).get("status_msg", "")
            print(f"  {test_model}: status={resp.status_code}, choices={choices is not None}, msg={status_msg[:50] if status_msg else 'OK'}")
            if choices:
                print(f"    WORKING! Response: {choices[0].get('message', {}).get('content', '')[:100]}...")
                break
        except Exception as e:
            print(f"  {test_model}: Error - {e}")
    
    try:
        answer = client.generate_answer(
            query="陕西电力市场的主要特点是什么",
            provincial_chunks=[],
            global_chunks=[],
            history=[],
            province_code="SN",
        )
        print(f"\nAnswer: {answer[:200]}...")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
else:
    print("Client not ready, skipping test")
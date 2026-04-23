#!/usr/bin/env python3
"""
Test LangChain LLM direct call
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from app.langchain.llm import MiniMaxLLMWrapper
import os
import time

api_key = os.getenv("LLM_API_KEY")
endpoint = os.getenv("LLM_ENDPOINT")
model = os.getenv("LLM_MODEL")

print(f"API Key: {api_key[:20]}...")
print(f"Endpoint: {endpoint}")
print(f"Model: {model}")

wrapper = MiniMaxLLMWrapper(
    api_key=api_key,
    endpoint=endpoint,
    model=model,
    disable_thinking=True,
)

system_prompt = "你是电力政策问答助手。"
user_content = "请用一句话介绍陕西电力市场。"

print("\nCalling MiniMax LLM...")
start = time.time()

try:
    answer = wrapper.invoke(user_content, system=system_prompt)
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    print(f"Answer: {answer}")
except Exception as e:
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    print(f"Error: {type(e).__name__}: {str(e)[:200]}")
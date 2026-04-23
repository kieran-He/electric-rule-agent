#!/usr/bin/env python3
"""
Test LangChain LLM direct call with full response output
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from app.langchain.llm import create_minimax_llm
from langchain_core.messages import HumanMessage, SystemMessage
import os
import time

api_key = os.getenv("LLM_API_KEY")
endpoint = os.getenv("LLM_ENDPOINT")
model = os.getenv("LLM_MODEL")

print(f"API Key: {api_key[:20]}...")
print(f"Endpoint: {endpoint}")
print(f"Model: {model}")

llm = create_minimax_llm(
    api_key=api_key,
    endpoint=endpoint,
    model=model,
)

system_prompt = "你是电力政策问答助手。"
user_content = "请用一句话介绍陕西电力市场。"

print("\nCalling MiniMax LLM...")
start = time.time()

messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]

try:
    response = llm.invoke(messages)
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    print(f"Response type: {type(response)}")
    print(f"Response content blocks: {len(response.content)}")
    
    for i, block in enumerate(response.content):
        print(f"\nBlock {i}: type={getattr(block, 'type', 'unknown')}")
        if hasattr(block, 'text'):
            print(f"  text: {block.text[:200]}")
        if hasattr(block, 'thinking'):
            print(f"  thinking: {block.thinking[:200]}")
        print(f"  block: {block}")
except Exception as e:
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    print(f"Error: {type(e).__name__}: {str(e)[:500]}")
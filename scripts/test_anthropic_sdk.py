#!/usr/bin/env python3
"""
Test MiniMax API with Anthropic SDK
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
import time
from app.generator import LLMClient

api_key = os.getenv('LLM_API_KEY', '')
endpoint = os.getenv('LLM_ENDPOINT', '')
model = os.getenv('LLM_MODEL', '')

print(f'API Key: {api_key[:20]}...')
print(f'Endpoint: {endpoint}')
print(f'Model: {model}')

client = LLMClient(
    api_key=api_key,
    endpoint=endpoint,
    model=model,
)

print(f'\nClient ready: {client.ready}')
print(f'Client mode: {client.mode}')

if client.ready:
    print('\nTesting LLMClient.generate_answer...')
    
    from app.repository import PolicyChunk
    
    # Create mock chunks for testing
    mock_chunks = [
        PolicyChunk(
            text='陕西电力市场是以煤电为基础、西电东送为特色的区域性电力交易枢纽。',
            metadata={'source_name': 'test_doc', 'title_path': '简介'},
            score=0.8,
        )
    ]
    
    start = time.time()
    try:
        answer = client.generate_answer(
            query='陕西电力市场的主要特点是什么？',
            provincial_chunks=mock_chunks,
            global_chunks=[],
            history=[],
            province_code='SN',
        )
        elapsed = time.time() - start
        
        print(f'Answer: {answer[:200]}...')
        print(f'\nTime: {elapsed:.2f}s')
        print('SUCCESS!')
        
    except Exception as e:
        print(f'Error: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
else:
    print('Client not ready, skipping test')
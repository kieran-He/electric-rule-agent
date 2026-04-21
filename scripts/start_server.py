#!/usr/bin/env python3
"""
Start API server with .env loaded
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from app.main import app

print("Starting server with .env loaded...")
print(f"LLM API Key: {__import__('os').getenv('LLM_API_KEY', '')[:15]}...")
print(f"LLM Endpoint: {__import__('os').getenv('LLM_ENDPOINT', '')}")
print(f"LLM Model: {__import__('os').getenv('LLM_MODEL', '')}")
print(f"Embedding Model: {__import__('os').getenv('EMBEDDING_MODEL', '')}")

uvicorn.run(app, host="localhost", port=8000)
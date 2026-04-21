"""Test query with deterministic embedder"""
import subprocess
import time
import requests

server_proc = subprocess.Popen(
    ["python", "-m", "uvicorn", "app.main:app", "--host", "localhost", "--port", "8000"],
)

time.sleep(3)

try:
    response = requests.post(
        "http://localhost:8000/query",
        json={"query": "陕西中长期签约比例", "session_id": "test", "province_codes": ["SN"]},
        timeout=30
    )
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Answer: {data.get('answer', '')[:200]}")
    print(f"Citations: {len(data.get('citations', []))}")
    print(f"Confidence: {data.get('confidence', 0)}")
    print(f"Used documents: {data.get('used_documents', [])}")
finally:
    server_proc.terminate()
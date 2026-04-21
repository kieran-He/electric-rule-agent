"""Start server and test in same process"""
import subprocess
import time
import requests
import json

server_proc = subprocess.Popen(
    ["python", "-m", "uvicorn", "app.main:app", "--host", "localhost", "--port", "8000"],
    cwd="E:/newprojects/firstmodel"
)

time.sleep(5)

try:
    response = requests.post(
        "http://localhost:8000/query",
        json={"query": "陕西中长期签约比例", "session_id": "test", "province_codes": ["SN"]},
        timeout=30
    )
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Answer (first 200 chars): {data.get('answer', '')[:200]}")
    print(f"Citations count: {len(data.get('citations', []))}")
    print(f"Confidence: {data.get('confidence', 0)}")
    print(f"Warnings: {data.get('warnings', [])}")
finally:
    server_proc.terminate()
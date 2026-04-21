import requests

try:
    resp = requests.get("http://localhost:8000/health", timeout=10)
    print(f"Health: {resp.status_code} - {resp.text}")
    
    resp = requests.post("http://localhost:8000/query", json={
        "query": "test",
        "session_id": "test001"
    }, timeout=60)
    print(f"Query: {resp.status_code}")
    print(f"Response: {resp.text[:1000]}")
except Exception as e:
    print(f"Error: {e}")
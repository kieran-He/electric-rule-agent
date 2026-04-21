import requests
import json

url = "http://localhost:8000/query"
payload = {
    "query": "陕西2026年中长期签约比例",
    "session_id": "test_001",
    "province_codes": ["SN"]
}

try:
    response = requests.post(url, json=payload, timeout=30)
    print(f"Status: {response.status_code}")
    print(f"Response text: {response.text[:500]}")
    if response.status_code == 200:
        print(f"Response JSON: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
except Exception as e:
    print(f"Error: {e}")
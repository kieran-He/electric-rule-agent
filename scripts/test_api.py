import requests
import json

url = "http://localhost:8000/query"
payload = {
    "query": "电力现货市场的购电交易流程是什么",
    "session_id": "test_001",
    "province_codes": ["SN"]
}

try:
    response = requests.post(url, json=payload, timeout=120)
    print(f"Status: {response.status_code}")
    print(f"Response text: {response.text[:500]}")
    if response.status_code == 200:
        print(f"Response JSON: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
except Exception as e:
    print(f"Error: {e}")
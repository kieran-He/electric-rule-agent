"""Debug retrieval from API"""
import requests
import json

url = "http://localhost:8000/query"
payload = {
    "query": "陕西中长期签约比例",
    "session_id": "debug_test",
    "province_codes": ["SN"],
    "top_k": 3
}

response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")
data = response.json()
print(f"Answer: {data.get('answer', '')[:200]}")
print(f"Citations: {len(data.get('citations', []))}")
print(f"Confidence: {data.get('confidence', 0)}")
print(f"Warnings: {data.get('warnings', [])}")
print(f"Used documents: {data.get('used_documents', [])}")
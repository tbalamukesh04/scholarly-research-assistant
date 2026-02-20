import requests
import json

url = "http://127.0.0.1:8000/query"
payload = {"query": "How does the attention mechanism work?", "top_k": 5, "mode": "strict"}

try:
    response = requests.post(url, json=payload)
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
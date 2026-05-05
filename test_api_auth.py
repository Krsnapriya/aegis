import os
from pathlib import Path
from dotenv import load_dotenv
import requests

# Mimic app.py setup
_repo = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_repo / ".env")

API_BASE = os.getenv("PLUTCHIK_API_URL", "http://localhost:8000")
API_KEY = os.getenv("PLUTCHIK_API_KEY")

print(f"API_BASE: {API_BASE}")
print(f"API_KEY length: {len(API_KEY) if API_KEY else 'None'}")

def test_explain():
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    payload = {"text": "Diagnostic test", "session_id": "diag"}
    try:
        response = requests.post(f"{API_BASE}/explain", json=payload, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Body: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_explain()

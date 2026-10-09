import urllib.request
import json
import sys

def test_all():
    print("=== 1. Testing Backend Server Status ===")
    try:
        with urllib.request.urlopen("http://127.0.0.1:5000/", timeout=5) as res:
            print(f"Root / Status: {res.status} OK")
    except Exception as e:
        print(f"Error connecting to Root: {e}")

    print("\n=== 2. Testing /api/analyze Endpoint ===")
    try:
        with urllib.request.urlopen("http://127.0.0.1:5000/api/analyze?ticker=USDINR=X&period=1y", timeout=15) as res:
            data = json.loads(res.read().decode())
            print(f"Analyze USD/INR Price: {data.get('latest_price')}, Action: {data.get('deep_ai', {}).get('action')}, Confidence: {data.get('deep_ai', {}).get('confidence')}%")
    except Exception as e:
        print(f"Analyze error: {e}")

    print("\n=== 3. Testing /api/chat NLP Responses ===")
    queries = [
        "sunn tu yeh check krke bta ki apn forex trade india me kr sakte hai ya nahi",
        "search latest news on USD",
        "calculate risk for 500 capital on USDINR",
        "trade execute kr USD/INR par"
    ]

    for q in queries:
        req = urllib.request.Request(
            "http://127.0.0.1:5000/api/chat",
            data=json.dumps({"message": q}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as res:
                resp = json.loads(res.read().decode())
                print(f"Query: '{q[:35]}...' -> Type: {resp.get('type')}")
                print(f"Snippet: {resp.get('text', '')[:100]}...\n")
        except Exception as e:
            print(f"Chat error on '{q}': {e}")

if __name__ == "__main__":
    test_all()

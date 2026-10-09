import urllib.request
import json

req = urllib.request.Request(
    "http://127.0.0.1:5000/api/chat",
    data=json.dumps({"message": "sunn tu yeh check krke bta ki apn forex trade india me kr sakte hai ya nahi"}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req) as res:
    data = json.loads(res.read().decode("utf-8"))
    print("SUCCESS: Received JSON from /api/chat:")
    print(data.get("text"))

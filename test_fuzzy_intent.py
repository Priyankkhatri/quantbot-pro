import urllib.request
import json

req = urllib.request.Request(
    "http://127.0.0.1:5000/api/chat",
    data=json.dumps({"message": "tradinbgview open kr and udhar jaakre check kr ki login hai account ya nahi"}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req) as res:
    data = json.loads(res.read().decode("utf-8"))
    print("MATCHED INTENT TYPE:", data.get("type"))
    print("RESPONSE TEXT:\n", data.get("text"))

#!/usr/bin/env python3
"""Exercise environment selection and real inference through the local HTTP API."""
from pathlib import Path
import json
from urllib.request import Request, urlopen
from gradio_client import Client

client = Client("http://127.0.0.1:7860", download_files=False, verbose=False)
updates = client.predict("reacher", api_name="/list_environment")
checkpoint, episode = updates[0]["value"], updates[1]["value"]
response = client.predict(checkpoint, episode, 3, api_name="/forecast")
record = response[3]
assert record["sample"]["environment"] == "reacher" and len(record["rows"]) == 5
downloads = []
for item in response[5]:
    with urlopen(Request(item["url"], method="HEAD"), timeout=30) as reply:
        assert reply.status == 200 and int(reply.headers["Content-Length"]) == item["size"]
    downloads.append({"filename": item["orig_name"], "bytes": item["size"], "http_status": 200})
assert {item["filename"] for item in downloads} == {"model.pt", "config.json"}
path = Path(__file__).resolve().parent / "validation/api.json"
path.write_text(json.dumps({"status": "passed", "check": "environment choices, real Reacher dim-appearance forecast, and checkpoint downloads via running HTTP API", "record": record, "downloads": downloads}, indent=2) + "\n")
print(json.dumps({"status": "passed", "checkpoint": checkpoint, "rows": len(record["rows"])}))

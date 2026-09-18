#!/usr/bin/env python3
"""Serve only the exported site on loopback; refresh status from local records."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from publish import HERE, refresh as build

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, default=8008)
parser.add_argument("--refresh-seconds", type=int, default=60)
args = parser.parse_args()
if args.refresh_seconds < 10: raise ValueError("Refresh interval must be at least 10 seconds")
build()
def refresh():
    while True:
        time.sleep(args.refresh_seconds)
        try: build()
        except Exception as error: print(f"Status refresh retained previous snapshot: {error}", flush=True)
threading.Thread(target=refresh, daemon=True).start()
server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(SimpleHTTPRequestHandler, directory=str(HERE)))
print(f"Local ShiftWM page: http://127.0.0.1:{args.port}", flush=True)
server.serve_forever()

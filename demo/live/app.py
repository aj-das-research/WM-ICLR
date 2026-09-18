"""Public read-only demonstration: fixed samples, bounded CPU inference."""
from collections import deque
import json
from pathlib import Path
import threading
import time
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
import backend

HERE = Path(__file__).resolve().parent
app = FastAPI(title="ShiftWM live checkpoint explorer", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["https://aj-das-research.github.io"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
ASSETS = {frame["file"]: HERE / "samples" / frame["file"] for row in backend.MANIFEST["samples"]
          for camera in row["cameras"].values() for frame in camera["frames"]}
STATIC = {"/": (HERE / "static/index.html", "text/html"),
          "/app.js": (HERE / "static/app.js", "text/javascript"),
          "/style.css": (HERE / "static/style.css", "text/css")}
CALL_TIMES = deque()
RATE_LOCK = threading.Lock()


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sample_id: str
    camera: Literal[1, 2] = 1
    seed: Literal[0, 1, 2] = 0
    horizon: Literal[5, 10] = 5


@app.middleware("http")
async def response_policy(request, call_next):
    size = request.headers.get("content-length", "0")
    if not size.isdigit() or int(size) > 2048:
        return JSONResponse({"detail": "Request too large"}, status_code=413)
    if request.method == "POST" and request.headers.get("transfer-encoding"):
        return JSONResponse({"detail": "Use a bounded JSON request"}, status_code=400)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'self' https://aj-das-research.github.io http://127.0.0.1:8047"
    return response


@app.get("/health")
def health():
    return JSONResponse({"status": "ready", "inference": "cpu", "samples": len(backend.SAMPLE_MAP),
            "input": "cached DINOv2 support features and recorded actions", "max_concurrent_inference": 1},
            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"})


@app.get("/api/catalog")
def catalog():
    return backend.MANIFEST


@app.post("/api/forecast")
def predict(body: ForecastRequest):
    with RATE_LOCK:
        now = time.monotonic()
        while CALL_TIMES and now - CALL_TIMES[0] >= 60:
            CALL_TIMES.popleft()
        if len(CALL_TIMES) >= 30:
            raise HTTPException(429, "The shared demo is busy; retry in one minute.")
        CALL_TIMES.append(now)
    try:
        return backend.forecast(**body.model_dump())
    except BlockingIOError as exc:
        raise HTTPException(429, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/assets/{filename}")
def asset(filename: str):
    if filename not in ASSETS:
        raise HTTPException(404, "Unknown asset")
    return FileResponse(ASSETS[filename], media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/{path:path}")
def static(path: str):
    key = "/" + path
    if key not in STATIC:
        raise HTTPException(404, "Unknown route")
    file, content_type = STATIC[key]
    return FileResponse(file, media_type=content_type)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7861, workers=1, limit_concurrency=16,
                timeout_keep_alive=5, access_log=False, log_level="info")

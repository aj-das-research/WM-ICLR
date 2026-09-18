#!/usr/bin/env python3
"""Exercise real Gradio inference using system Firefox and WebDriver HTTP.

Browser/IAB tools are unavailable in this session. Firefox and the official
Mozilla geckodriver avoid downloading a second large browser distribution.
"""
import base64
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:4444"


def call(method, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(BASE + route, data=data, method=method, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=90) as response:
        return json.load(response)["value"]


def main():
    log = (HERE / "validation/geckodriver.log").open("w")
    process = subprocess.Popen([str(HERE / "tools/geckodriver"), "--host", "127.0.0.1", "--port", "4444"], stdout=log, stderr=subprocess.STDOUT)
    session = None
    try:
        for _ in range(30):
            try:
                call("GET", "/status"); break
            except OSError:
                time.sleep(.3)
        session = call("POST", "/session", {"capabilities": {"alwaysMatch": {"browserName": "firefox", "moz:firefoxOptions": {"args": ["-headless"]}}}})["sessionId"]
        route = f"/session/{session}"
        call("POST", route + "/window/rect", {"width": 1366, "height": 1100})
        call("POST", route + "/url", {"url": "http://127.0.0.1:7860"})
        for _ in range(60):
            text = call("POST", route + "/execute/sync", {"script": "return document.body.innerText", "args": []})
            if "Run real checkpoint inference" in text:
                break
            time.sleep(.5)
        button = call("POST", route + "/element", {"using": "xpath", "value": "//button[contains(., 'Run real checkpoint inference')]"})
        button_id = button["element-6066-11e4-a52e-4f735466cecf"]
        call("POST", route + f"/element/{button_id}/click", {})
        for _ in range(90):
            text = call("POST", route + "/execute/sync", {"script": "return document.body.innerText", "args": []})
            if "These five forecasts are a single example" in text:
                break
            if "Error" in text and "Traceback" in text:
                raise RuntimeError(text)
            time.sleep(1)
        else:
            raise RuntimeError("Browser inference did not finish")
        (HERE / "validation/desktop.png").write_bytes(base64.b64decode(call("GET", route + "/moz/screenshot/full")))
        desktop = call("POST", route + "/execute/sync", {"script": "return {width:innerWidth,scrollWidth:document.documentElement.scrollWidth,images:document.querySelectorAll('img').length}", "args": []})
        call("POST", route + "/window/rect", {"width": 390, "height": 844})
        time.sleep(1)
        (HERE / "validation/mobile.png").write_bytes(base64.b64decode(call("GET", route + "/moz/screenshot/full")))
        mobile = call("POST", route + "/execute/sync", {"script": "return {width:innerWidth,scrollWidth:document.documentElement.scrollWidth}", "args": []})
        if mobile["scrollWidth"] > mobile["width"] + 2:
            raise RuntimeError(f"Mobile page overflow: {mobile}")
        evidence = {"status": "passed", "browser": "system Firefox via official Mozilla geckodriver0.36.0", "desktop": desktop, "mobile": mobile,
                    "checks": ["local app loaded", "actual inference button clicked", "real computed response visible", "desktop and mobile screenshots", "no page-level mobile overflow"],
                    "visible_text": text}
        (HERE / "validation/browser.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps({"status": "passed", "desktop": desktop, "mobile": mobile}))
    finally:
        if session:
            call("DELETE", f"/session/{session}")
        process.terminate()
        process.wait(timeout=10)
        log.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Real Firefox validation when Browser/IAB and a Chromium installation are absent."""
import base64
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:4445"

def call(method, path, payload=None):
    with urlopen(Request(BASE + path, data=None if payload is None else json.dumps(payload).encode(),
                         method=method, headers={"Content-Type": "application/json"}), timeout=60) as reply:
        return json.load(reply)["value"]

log = (HERE / "validation/browser.log").open("w")
process = subprocess.Popen([str(HERE.parent / "demo/tools/geckodriver"), "--port", "4445", "--host", "127.0.0.1"], stdout=log, stderr=subprocess.STDOUT)
session = None
try:
    for _ in range(20):
        try: call("GET", "/status"); break
        except OSError: time.sleep(.2)
    session = call("POST", "/session", {"capabilities": {"alwaysMatch": {"browserName": "firefox", "moz:firefoxOptions": {"args": ["-headless"]}}}})["sessionId"]
    route = f"/session/{session}"
    def script(value): return call("POST", route + "/execute/sync", {"script": value, "args": []})
    def click(selector):
        item = call("POST", route + "/element", {"using": "css selector", "value": selector})
        call("POST", route + "/element/" + item["element-6066-11e4-a52e-4f735466cecf"] + "/click", {})
    def screenshot(name): (HERE / "validation" / name).write_bytes(base64.b64decode(call("GET", route + "/screenshot")))
    call("POST", route + "/window/rect", {"width": 1440, "height": 1100})
    call("POST", route + "/url", {"url": "http://127.0.0.1:8008"})
    for _ in range(40):
        if "Loading" not in script("return document.querySelector('#training-status').textContent"): break
        time.sleep(.25)
    assert not script("return !document.querySelector('#status-error').hidden")
    assert script("return [...document.images].every(i=>i.complete&&i.naturalWidth>0)")
    above_fold = script("return [...document.querySelectorAll('.site-header a,.hero h1,.hero-subtitle,.hero-description,.actions a,.status-note')].map(e=>e.textContent.trim())")
    screenshot("desktop-hero.png")
    script("document.getElementById('method').scrollIntoView({behavior:'instant'})")
    screenshot("desktop-method.png")
    click("nav a[href='#progress']")
    time.sleep(.5)
    click("#refresh-status")
    time.sleep(.6)
    screenshot("desktop-progress.png")
    click("#copy-command")
    assert script("return document.querySelector('#copy-command').textContent") == "Copied"
    click(".protocol-details summary")
    assert script("return document.querySelector('.protocol-details').open")
    assert script("return document.documentElement.scrollWidth<=innerWidth")
    body = script("return document.body.innerText")
    desktop = script("return {width:innerWidth,height:innerHeight,h1:getComputedStyle(document.querySelector('h1')).fontSize}")
    # A same-origin iframe supplies a real390px layout viewport despite Firefox's500px minimum outer window.
    script("document.body.innerHTML='<iframe id=mobile-frame src=/index.html style=\"display:block;width:390px;height:1000px;border:0\"></iframe>';document.body.style.margin='0'")
    time.sleep(1)
    mobile = script("const f=document.querySelector('iframe');return {width:f.contentWindow.innerWidth,scrollWidth:f.contentDocument.documentElement.scrollWidth}")
    assert mobile["width"] == 390 and mobile["scrollWidth"] <= 390, mobile
    script("const f=document.querySelector('iframe');f.style.height=f.contentDocument.documentElement.scrollHeight+'px'")
    frame = call("POST", route + "/element", {"using": "css selector", "value": "#mobile-frame"})
    frame_id = frame["element-6066-11e4-a52e-4f735466cecf"]
    (HERE / "validation/mobile.png").write_bytes(base64.b64decode(call("GET", route + f"/element/{frame_id}/screenshot")))
    evidence = {"status": "passed", "method": "system Firefox152 / official GeckoDriver0.36; Browser/IAB unavailable and Chromium not installed",
                "desktop": desktop, "mobile": mobile, "above_fold_copy": above_fold,
                "checks": ["real status JSON rendered", "no status error", "actual method SVG loaded", "navigation links", "refresh status", "clipboard feedback", "protocol disclosure", "no desktop or390px overflow"],
                "visible_text": body}
    (HERE / "validation/browser.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"status": "passed", "desktop": desktop, "mobile": mobile}))
finally:
    if session: call("DELETE", f"/session/{session}")
    process.terminate(); process.wait(timeout=10); log.close()

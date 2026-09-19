#!/usr/bin/env python3
"""Export native draw.io through its official browser application.

Uses an isolated Playwright environment and an existing Chrome installation.
The editor receives only the requested diagram. Browser work is bounded and
its private process group is closed even if the browser's shutdown stalls.
Protocol: https://www.drawio.com/docs/reference/embed-mode/
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ET


def worker(source: Path, prefix: Path) -> None:
    from playwright.sync_api import sync_playwright
    source_bytes = source.read_bytes()
    xml = source_bytes.decode()
    ET.fromstring(xml)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shiftwm-diagram-") as td:
        wrapper = Path(td) / "host.html"
        payload = json.dumps(xml).replace("<", "\\u003c")
        wrapper.write_text("""<!doctype html><meta charset="utf-8">
<style>body{margin:0}iframe{width:1600px;height:1100px;border:0}</style>
<iframe id="editor"></iframe><script>
const xml=""" + payload + """;
window.exports=[];window.events=[];
const frame=document.getElementById('editor');
window.addEventListener('message',e=>{
 if(e.origin!=='https://embed.diagrams.net')return;
 let m;try{m=typeof e.data==='string'?JSON.parse(e.data):e.data}catch{return}
 window.events.push({event:m.event,error:m.error});
 const send=obj=>e.source.postMessage(JSON.stringify(obj),e.origin);
 if(m.event==='init')send({action:'load',xml,fit:1,border:0,background:'#ffffff'});
 if(m.event==='load')send({action:'export',format:'svg',border:0,background:'#ffffff',embedImages:true});
 if(m.event==='export'){
  window.exports.push(m);
  if(window.exports.length===1)send({action:'export',format:'png',scale:4,width:1650,border:0,background:'#ffffff'});
 }
});
frame.src='https://embed.diagrams.net/?embed=1&proto=json&spin=1&ui=min';
</script>""")
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path="/usr/bin/google-chrome",
                                        headless=True,
                                        args=["--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": 1600, "height": 1100})
            page.goto(wrapper.as_uri(), wait_until="domcontentloaded", timeout=30000)
            page.wait_for_function("window.exports.length===2", timeout=60000)
            records = []
            for obj in page.evaluate("window.exports"):
                header, payload = obj["data"].split(",", 1)
                raw = (base64.b64decode(payload) if ";base64" in header
                       else urllib.parse.unquote_to_bytes(payload))
                suffix = obj["format"]
                if suffix not in {"svg", "png"}:
                    raise ValueError("Unexpected native export format")
                if suffix == "svg":
                    ET.fromstring(raw)
                output = Path(str(prefix) + "." + suffix)
                output.write_bytes(raw)
                records.append({"path": str(output), "sha256": hashlib.sha256(raw).hexdigest(),
                                "bytes": len(raw), "bounds": obj.get("bounds")})
            page.screenshot(path=str(prefix) + ".browser.png")
            receipt = {"status": "native_browser_export_complete", "source": str(source),
                       "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                       "editor": "https://embed.diagrams.net/", "exports": records,
                       "events": page.evaluate("window.events"),
                       "scope": "Actual native application rendering; visual review is a separate step."}
            Path(str(prefix) + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps({"status": receipt["status"], "output_prefix": str(prefix)}), flush=True)
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    source, prefix = args.source.resolve(), args.output_prefix.resolve()
    if args.worker:
        worker(source, prefix)
        return
    command = [sys.executable, str(Path(__file__).resolve()), str(source), str(prefix), "--worker"]
    expected_source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    started_ns = time.time_ns()
    process = subprocess.Popen(command, start_new_session=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    record_path = Path(str(prefix) + ".json")
    deadline = time.monotonic() + 105
    ready = False
    while process.poll() is None and time.monotonic() < deadline:
        if record_path.exists() and record_path.stat().st_mtime_ns >= started_ns:
            try:
                record = json.loads(record_path.read_text())
                ready = record.get("source_sha256") == expected_source_sha
            except json.JSONDecodeError:
                pass
            if ready:
                break
        time.sleep(0.2)
    try:
        output, _ = process.communicate(timeout=8 if ready else 1)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        if not record_path.exists() or record_path.stat().st_mtime_ns < started_ns:
            raise RuntimeError("Native export timed out before producing a receipt")
        record = json.loads(record_path.read_text())
        if record["source_sha256"] != expected_source_sha:
            raise RuntimeError("Stale native export receipt")
        record["browser_shutdown"] = "Dedicated process group closed after export completed"
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    else:
        if process.returncode:
            raise RuntimeError(output[-3000:])
    print(output.strip())


if __name__ == "__main__":
    main()

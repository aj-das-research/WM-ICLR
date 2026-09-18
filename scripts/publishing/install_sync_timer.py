#!/usr/bin/env python3
"""Install the authorized publishing bridge as a five-minute user timer."""
from pathlib import Path
import argparse
import json
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=Path.home() / ".config/shiftwm/publishing.json")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    state = Path(settings.get("sync_state", Path.home() / ".local/share/shiftwm/sync"))
    if not (state / "state.json").is_file():
        raise SystemExit("Run sync_project.py --initialize and a verified full sync first")
    if not (state / "receipt.json").is_file():
        raise SystemExit("Complete one successful manual synchronization before enabling the timer")
    units = Path.home() / ".config/systemd/user"
    units.mkdir(parents=True, exist_ok=True)
    quote = lambda value: shlex.quote(str(value)).replace("%", "%%")
    service = f"""[Unit]
Description=Synchronize ShiftWM workspace, GitHub, Overleaf and project page

[Service]
Type=oneshot
WorkingDirectory={quote(ROOT)}
Environment=PATH={Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin
ExecStart={quote(sys.executable)} {quote(ROOT / 'scripts/publishing/sync_project.py')} --config {quote(args.config.resolve())} --watch-cycle
TimeoutStartSec=20min
UMask=0077
Nice=10
StandardOutput=append:{state / 'timer.log'}
StandardError=inherit
"""
    timer = """[Unit]
Description=Check ShiftWM publishing synchronization every five minutes

[Timer]
OnStartupSec=2min
OnUnitInactiveSec=5min
AccuracySec=15s
Unit=shiftwm-sync.service

[Install]
WantedBy=timers.target
"""
    (units / "shiftwm-sync.service").write_text(service)
    (units / "shiftwm-sync.timer").write_text(timer)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", "shiftwm-sync.timer"], check=True)
    subprocess.run(["systemctl", "--user", "list-timers", "shiftwm-sync.timer", "--no-pager"], check=True)


if __name__ == "__main__":
    main()

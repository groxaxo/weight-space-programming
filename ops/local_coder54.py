#!/usr/bin/env python3
"""Start/stop the optional local coding endpoint on .54 with Bonsai restoration."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from coder_common import check, loads

UNIT = "wsp54-local-coder.service"


def status():
    state = subprocess.run(["systemctl", "--user", "is-active", UNIT], capture_output=True, text=True)
    result = {"service": UNIT, "state": state.stdout.strip() or "unknown"}
    if state.returncode == 0:
        try:
            with urlopen("http://127.0.0.1:18084/health", timeout=3) as response:
                result["endpoint"] = loads(response.read())
        except OSError:
            result["endpoint"] = "starting; inspect journalctl --user -u " + UNIT
    result["bonsai"] = subprocess.run(["systemctl", "--user", "is-active", "browserfusion-bonsai.service"],
                                      capture_output=True, text=True).stdout.strip()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "status"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--adapter", type=Path, help="A complete, evaluated adapter directory")
    selection.add_argument("--base", action="store_true", help="Serve the unchanged base model for comparison")
    parser.add_argument("--allow-pause-bonsai", action="store_true")
    args = parser.parse_args()
    if args.action != "status":
        from method_policy import blocked_legacy_operation
        blocked_legacy_operation("Legacy service start/stop launcher")
    check(os.uname().nodename == "4790", "This launcher is for .54 only")
    if args.action == "start":
        check(args.base or (args.adapter and args.adapter.is_dir()), "Supply --base or an evaluated --adapter directory")
        check(args.allow_pause_bonsai, "Starting this endpoint requires --allow-pause-bonsai")
        check(subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT]).returncode != 0,
              "Local coder is already running")
        output = ROOT / "runs" / ("serve-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        supervisor = str(ROOT / "ops/run54_gpu_phase.py")
        command = ["systemd-run", "--user", "--collect", "--unit=" + UNIT, "--property=RuntimeMaxSec=7200",
                   "--property=TimeoutStopSec=180", "--property=KillMode=control-group",
                   "--property=WorkingDirectory=" + str(ROOT),
                   "--property=ExecStopPost=/usr/bin/python3 " + supervisor + " --restore-only --output " + str(output),
                   "/usr/bin/python3", supervisor, "--phase", "serve", "--output", str(output), "--allow-pause-bonsai"]
        if args.adapter:
            command.extend(["--adapter", str(args.adapter.resolve())])
        subprocess.run(command, check=True)
        print(json.dumps({"status": "starting", "output": str(output), "url": "http://127.0.0.1:18084/v1",
                          "maximum_session_seconds": 7200, "stop": "python3 ops/local_coder54.py stop"}))
    elif args.action == "stop":
        subprocess.run(["systemctl", "--user", "stop", UNIT], check=True, timeout=210)
        print(json.dumps(status()))
    else:
        print(json.dumps(status()))


if __name__ == "__main__":
    main()

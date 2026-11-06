#!/usr/bin/env python3
"""Run one supervised .54 GPU phase and restore the pre-existing Bonsai service.

The caller must explicitly authorize --allow-pause-bonsai. No persistent service
configuration is changed. Use a transient systemd unit with --restore-only as
ExecStopPost for restoration even when this process is forcibly terminated.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from coder_common import check, load_config, loads, write_json

SERVICE = "browserfusion-bonsai.service"


def restore(output):
    marker = output / "restore-bonsai-required"
    before_path = output / "service-before-phase.json"
    before = loads(before_path.read_text()) if before_path.exists() else {}
    if before:
        check(before.get("service") == SERVICE, "Unexpected service restoration record")
    if marker.exists() or before.get("was_active") is True:
        subprocess.run(["systemctl", "--user", "start", SERVICE], check=True, timeout=120)
        subprocess.run(["systemctl", "--user", "is-active", "--quiet", SERVICE], check=True, timeout=15)
        marker.unlink(missing_ok=True)
        path = output / "bonsai-restored.json"
        if not path.exists():
            write_json(path, {"status": "active", "restored_at_utc": datetime.now(timezone.utc).isoformat()})


def stop_child(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@contextmanager
def gpu_window(output, allow_pause):
    check(os.uname().nodename == "4790", "This supervisor is for .54 only")
    with (ROOT / "runs/.gpu-phase.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = subprocess.run(["systemctl", "--user", "is-active", "--quiet", SERVICE])
        check(observed.returncode in (0, 3), "Cannot determine Bonsai's initial service state")
        active = observed.returncode == 0
        check(not active or allow_pause, "Pausing the existing Bonsai service needs explicit authorization")
        write_json(output / "service-before-phase.json", {"service": SERVICE, "was_active": active,
            "query_exit_code": observed.returncode, "observed_at_utc": datetime.now(timezone.utc).isoformat()})
        try:
            if active:
                (output / "restore-bonsai-required").write_text(SERVICE + "\n")
                subprocess.run(["systemctl", "--user", "stop", SERVICE], check=True, timeout=90)
            yield
        finally:
            restore(output)


def run_logged(command, output, name, timeout, accepted_codes=(0,)):
    with (output / name).open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=timeout)
        finally:
            stop_child(process)
    check(code in accepted_codes, f"Phase command failed ({code}); inspect {output / name}")


def serve_and_evaluate(args, output, cfg, snapshot):
    command = [args.python, "-u", "scripts/serve_coder.py", "--config", str(args.config),
               "--model-path", snapshot, "--model-proof", str(args.model_proof),
               "--four-bit", "--direct-idle-gpu", "--max-context", "16384", "--port", str(args.port)]
    if args.adapter:
        command.extend(["--adapter", str(args.adapter)])
    with (output / "server.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while True:
                check(process.poll() is None, "Model server exited during startup; inspect server.log")
                try:
                    with urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                        info = loads(response.read())
                    check(info["wsp"]["revision"] == cfg["revision"], "Unexpected server identity")
                    check(info["wsp"]["adapter"] == (str(args.adapter) if args.adapter else None),
                          "Unexpected adapter on the serving port")
                    write_json(output / "server-ready.json", info)
                    break
                except OSError:
                    check(time.monotonic() < deadline, "Model server startup timed out")
                    time.sleep(2)
            if args.phase == "serve":
                check(process.wait() == 0, "Model server exited unexpectedly")
                return
            if args.phase != "benchmark":
                run_logged([args.python, "-u", "scripts/evaluate_coder.py", "--base-url", f"http://127.0.0.1:{args.port}/v1",
                    "--model", cfg["served_model_name"], "--image", args.image, "--output", str(output / "evaluation.json")],
                    output, "evaluation.log", 1800)
            if args.clients:
                run_logged([args.python, "-u", "ops/smoke_clients.py", "--base-url", f"http://127.0.0.1:{args.port}/v1",
                    "--model", cfg["served_model_name"], "--image", args.image, "--output", str(output / "clients")],
                    output, "clients.log", 1000, accepted_codes=(0, 2))
            if args.humaneval or args.phase == "benchmark":
                run_logged([args.python, "-u", "scripts/evaluate_humaneval.py", "--base-url", f"http://127.0.0.1:{args.port}/v1",
                    "--model", cfg["served_model_name"], "--image", args.image, "--output", str(output / "humaneval.json")],
                    output, "humaneval.log", 10800)
        finally:
            stop_child(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("evaluate", "benchmark", "smoke", "train", "serve"), default="evaluate")
    parser.add_argument("--python", default=str(ROOT / ".venv-train/bin/python"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen35-4b.json")
    parser.add_argument("--model-proof", type=Path, default=ROOT / "reports/model-inputs.json")
    parser.add_argument("--data", type=Path, default=ROOT / "assets/coder-corpus-v1/prepared")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--image", default="python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54")
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--allow-pause-bonsai", action="store_true")
    parser.add_argument("--clients", action="store_true")
    parser.add_argument("--humaneval", action="store_true")
    parser.add_argument("--restore-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.absolute()
    if args.restore_only:
        restore(args.output)
        return
    check(not args.output.exists(), "Refusing existing phase output")
    check(not args.adapter or args.phase in ("evaluate", "benchmark", "serve"), "Adapter is only valid for inference")
    args.output.mkdir(parents=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    cfg = load_config(args.config)
    snapshot = loads(args.model_proof.read_text())["snapshot"]
    write_json(args.output / "phase.json", {"phase": args.phase, "source": str(ROOT),
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "model_revision": cfg["revision"],
        "quantization": "nf4-double", "adapter": str(args.adapter) if args.adapter else None})
    def terminate(_signal, _frame):
        raise SystemExit(128 + _signal)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
                      PYTORCH_ALLOC_CONF="expandable_segments:True")
    try:
        with gpu_window(args.output, args.allow_pause_bonsai):
            if args.phase in ("evaluate", "benchmark", "serve"):
                serve_and_evaluate(args, args.output, cfg, snapshot)
            else:
                command = [args.python, "-u", "scripts/train_coder.py", "--config", str(args.config), "--data", str(args.data),
                    "--model-path", snapshot, "--model-proof", str(args.model_proof), "--four-bit", "--direct-idle-gpu",
                    "--output", str(args.output / "training")]
                if args.phase == "smoke":
                    command.extend(["--max-steps", "2"])
                run_logged(command, args.output, "training.log", 10800)
    except SystemExit as error:
        if args.phase == "serve" and error.code in (130, 143):
            write_json(args.output / "phase-completed.json", {"status": "stopped", "phase": args.phase,
                       "bonsai_restoration_pending": (args.output / "restore-bonsai-required").exists()})
            return
        write_json(args.output / "phase-failed.json", {"status": "failed", "phase": args.phase,
                   "error": f"{type(error).__name__}: {error}", "restoration_pending":
                   (args.output / "restore-bonsai-required").exists()})
        raise
    except BaseException as error:
        write_json(args.output / "phase-failed.json", {"status": "failed", "phase": args.phase,
                   "error": f"{type(error).__name__}: {error}", "restoration_pending":
                   (args.output / "restore-bonsai-required").exists()})
        raise
    write_json(args.output / "phase-completed.json", {"status": "complete", "phase": args.phase})


if __name__ == "__main__":
    main()

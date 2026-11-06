#!/usr/bin/env python3
"""Run installed OpenCode and Hermes against isolated file-edit fixtures.

Clients receive file tools only. The harness independently executes the result
inside the same pinned Docker sandbox used by the development evaluation.
No user's agent profile, provider defaults, or project files are modified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from coder_common import check, digest, write_json
from evaluate_coder import Client, run_python

INITIAL = '''def wsp_client_discount(price, percent):
    """Apply a percentage discount. Negative prices and percentages outside 0..100 are invalid."""
    return price * (1 + percent / 100)
'''
CHECKS = '''import math
assert math.isclose(wsp_client_discount(100, 25), 75)
assert math.isclose(wsp_client_discount(13.5, 100), 0)
assert math.isclose(wsp_client_discount(20, 0), 20)
assert math.isclose(wsp_client_discount(0, 30), 0)
for price, percent in [(-1, 20), (10, -1), (10, 101)]:
    try:
        wsp_client_discount(price, percent)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid input accepted')
'''
TEST_FILE = '''import unittest
from calc import wsp_client_discount
class DiscountTests(unittest.TestCase):
    def test_discount(self): self.assertAlmostEqual(wsp_client_discount(100, 25), 75)
    def test_invalid(self):
        with self.assertRaises(ValueError): wsp_client_discount(-1, 20)
if __name__ == '__main__': unittest.main()
'''
PROMPT = ("Read calc.py and test_calc.py. Fix wsp_client_discount(price, percent) so it applies a percentage discount "
          "and raises ValueError for negative prices or percentages outside 0..100. Preserve the function signature. "
          "Edit only calc.py. Use file tools to make the change. The independent harness will run the tests; "
          "do not execute commands or claim that you ran tests. Briefly explain the change.")


def execute(command, cwd, env, log_path, timeout=420):
    start = time.monotonic()
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            return {"exit_code": code, "timed_out": False, "seconds": time.monotonic() - start}
        except subprocess.TimeoutExpired:
            return {"exit_code": None, "timed_out": True, "seconds": time.monotonic() - start}
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18084/v1")
    parser.add_argument("--model", default="wsp-qwen35-4b-coder")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), "Refusing existing client test output")
    client = Client(args.base_url, args.model)
    args.output.mkdir(parents=True)
    results = []
    for name in ("opencode", "hermes"):
        work = args.output / name
        work.mkdir()
        (work / "calc.py").write_text(INITIAL)
        (work / "test_calc.py").write_text(TEST_FILE)
        (work / "prompt.txt").write_text(PROMPT)
        (work / "AGENTS.md").write_text("Work only in this fixture. Edit calc.py using file tools. The harness executes tests.\n")
        initial_hash = digest(work / "calc.py")
        tests_hash = digest(work / "test_calc.py")
        state = work / ".client-state"
        state.mkdir()
        env = {"PATH": "/home/op/.local/bin:/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "TERM": "dumb",
               "XDG_CONFIG_HOME": str(state / "config"), "XDG_DATA_HOME": str(state / "data"),
               "XDG_STATE_HOME": str(state / "state"), "XDG_CACHE_HOME": str(state / "cache")}
        executable = shutil.which(name) or f"/home/op/.local/bin/{name}"
        row = {"client": name, "passed": False, "scope": "isolated file tools; independent sandbox tests"}
        try:
            check(Path(executable).exists(), f"{name} is not installed")
            version = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=30)
            row["version"] = version.stdout.splitlines()[0] if version.stdout else "unknown"
            if name == "opencode":
                config = {"$schema": "https://opencode.ai/config.json", "model": f"wsp-local/{args.model}",
                    "provider": {"wsp-local": {"npm": "@ai-sdk/openai-compatible", "name": "WSP local trial",
                    "options": {"baseURL": args.base_url, "apiKey": "local-only"},
                    "models": {args.model: {"name": "Qwen 4B trial", "toolcall": True,
                        "limit": {"context": 16384, "output": 2048}}}}},
                    "tools": {"bash": False, "task": False, "webfetch": False, "websearch": False},
                    "permission": {"*": "deny", "read": "allow", "edit": "allow", "write": "allow",
                                   "glob": "allow", "grep": "allow", "list": "allow", "external_directory": "deny"}}
                write_json(work / "opencode.json", config)
                env["OPENCODE_CONFIG"] = str(work / "opencode.json")
                env["OPENCODE_CONFIG_DIR"] = str(state / "opencode-config")
                command = [executable, "run", "--pure", "--format", "json", "--model", f"wsp-local/{args.model}", PROMPT]
            else:
                env["HERMES_HOME"] = str(state / "hermes")
                command = [executable, "chat", "--safe-mode", "--provider", "custom", "--base-url", args.base_url,
                    "--api-key", "local-only", "--model", args.model, "--oneshot", "--format", "stream-json",
                    "--toolsets", "file", "--max-turns", "8", "--query-file", str(work / "prompt.txt")]
            row["command_argv"] = command
            row.update(execute(command, work, env, work / "client.log"))
            check((work / "calc.py").is_file() and not (work / "calc.py").is_symlink(), "Client did not leave a regular source file")
            check(digest(work / "test_calc.py") == tests_hash, "Client modified the fixture tests")
            check(digest(work / "calc.py") != initial_hash, "Client made no source edit")
            row["independent_tests"] = run_python((work / "calc.py").read_text(), CHECKS, args.image)
            row["source_after_sha256"] = digest(work / "calc.py")
            row["tests_sha256"] = tests_hash
            row["passed"] = row["exit_code"] == 0 and row["independent_tests"]["passed"]
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        results.append(row)
        print(json.dumps(row), flush=True)
    report = {"schema_version": 1, "identity": client.identity, "results": results,
              "passed": all(row["passed"] for row in results), "sandbox_image": args.image,
              "harness_sha256": digest(Path(__file__))}
    write_json(args.output / "acceptance.json", report)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()

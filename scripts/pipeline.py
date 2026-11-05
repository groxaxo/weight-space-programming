#!/usr/bin/env python3
"""Pinned, fail-closed reproduction of the published Qwen3.5-0.8B edit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REV = "2fc06364715b967f1860aea9cf38778875588b17"
UP = ROOT / "upstream/lm_understand-logs/exp8_math"
EDIT = UP / "runs/p50c/edit"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def preflight():
    assert (UP / "code/apply_edit.py").is_file(), "Missing upstream research repository"
    assert (EDIT / "manifest.json").is_file() and (EDIT / "edit.npz").is_file(), "Missing published edit"
    manifest = json.loads((EDIT / "manifest.json").read_text())
    assert manifest["tensors"], "Empty manifest"
    print(json.dumps({"revision": REV, "edited_tensors": len(manifest["tensors"]), "edit_sha256": sha256(EDIT / "edit.npz"), "disk_free_gib": round(os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize / 2**30, 2)}, indent=2))

def download():
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id="Qwen/Qwen3.5-0.8B", revision=REV, local_dir=str(ROOT / "models/base"))
    print("Downloaded pinned base:", path)

def rebuild(base, output):
    assert base.is_file(), f"Base safetensors not found: {base}"
    assert output.suffix == ".safetensors", "Output must be a safetensors file"
    assert base.resolve() != output.resolve(), "Refusing to overwrite original model"
    assert not output.exists(), "Output already exists; refusing overwrite"
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(UP / "code/apply_edit.py"), str(base), str(EDIT), str(output)], check=True)
    report = {"base": str(base), "base_sha256": sha256(base), "output": str(output), "output_sha256": sha256(output), "edit_sha256": sha256(EDIT / "edit.npz"), "revision": REV}
    (ROOT / "reports/rebuild.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["preflight", "download", "rebuild"])
    p.add_argument("--base", type=Path, default=ROOT / "models/base/model.safetensors")
    p.add_argument("--output", type=Path, default=ROOT / "models/rebuilt/model.safetensors")
    args = p.parse_args()
    preflight()
    if args.action == "download": download()
    if args.action == "rebuild": rebuild(args.base, args.output)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download the pinned official snapshot and authenticate Hub blob/LFS identities."""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
from coder_common import check, digest, load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/qwen35-4b.json")
    parser.add_argument("--proof", type=Path, required=True)
    args = parser.parse_args()
    check(not args.proof.exists() and not args.proof.is_symlink(), "Refusing existing model proof")
    from huggingface_hub import HfApi, snapshot_download
    cfg = load_config(args.config)
    info = HfApi().model_info(cfg["model_id"], revision=cfg["revision"], files_metadata=True)
    check(info.sha == cfg["revision"], "Hub returned a different model revision")
    patterns = ["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE"]
    files = [f for f in info.siblings if any(fnmatch.fnmatch(f.rfilename, pattern) for pattern in patterns)]
    check(all(Path(f.rfilename).name == f.rfilename for f in files), "Unexpected nested model file")
    root = Path(snapshot_download(cfg["model_id"], revision=cfg["revision"], allow_patterns=patterns, max_workers=2))
    proof = {"schema_version": 1, "model_id": cfg["model_id"], "revision": cfg["revision"],
             "snapshot": str(root), "files": {}}
    for entry in files:
        path = root / entry.rfilename
        check(path.stat().st_size == entry.size, f"Model file size mismatch: {entry.rfilename}")
        sha = digest(path)
        if entry.lfs:
            check(sha == entry.lfs.sha256, f"Model LFS hash mismatch: {entry.rfilename}")
        else:
            blob = path.read_bytes()
            git_hash = hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest()
            check(git_hash == entry.blob_id, f"Model Git blob mismatch: {entry.rfilename}")
        proof["files"][entry.rfilename] = {"size": entry.size, "sha256": sha,
            "upstream_identity": entry.lfs.sha256 if entry.lfs else entry.blob_id}
    write_json(args.proof, proof)
    print(json.dumps({"status": "model_authenticated", "snapshot": str(root), "files": len(files), "proof": str(args.proof)}), flush=True)


if __name__ == "__main__":
    main()

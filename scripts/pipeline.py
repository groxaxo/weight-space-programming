#!/usr/bin/env python3
"""CPU-only, locked reproduction of the published p50c sparse BF16 repair.

No upstream Python is executed. The repository lock file is the trust anchor.
This is a reproduction tool, not a sandbox for autonomous model-generated edits.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 4 * 1024 * 1024
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8,
               "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


class ValidationError(RuntimeError):
    """An input, integrity, or non-destructive-publication check failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    result = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    require(isinstance(result, dict), f"Expected JSON object: {path}")
    return result


def leaf(name: str) -> str:
    require(isinstance(name, str) and name not in ("", ".", "..")
            and Path(name).name == name and not any(c in name for c in "\\/\x00"),
            f"Unsafe filename: {name!r}")
    return name


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(path: Path, spec: dict) -> None:
    require(path.is_file() and not path.is_symlink(), f"Missing regular file: {path}")
    require(path.stat().st_size == spec["size"], f"Size mismatch: {path}")
    require(sha256(path) == spec["sha256"], f"SHA-256 mismatch: {path}")


def load_lock(path: Path) -> dict:
    lock = read_json(path)
    require(lock.get("schema_version") == 1, "Unsupported reproduction lock schema")
    require(set(lock["source"]["files"]) == {"manifest.json", "edit.npz"},
            "Lock must contain exactly the manifest and edit archive")
    for group in (lock["source"]["files"], lock["model"]["files"]):
        for name, spec in group.items():
            leaf(name)
            require(type(spec["size"]) is int and spec["size"] > 0, "Invalid locked size")
            h = spec["sha256"]
            require(isinstance(h, str) and len(h) == 64
                    and all(c in "0123456789abcdef" for c in h), "Invalid SHA-256")
    for section in ("source", "model"):
        revision = lock[section]["revision"]
        require(len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
                f"Expected full immutable revision for {section}")
    return lock


def fetch_file(url: str, destination: Path, spec: dict) -> None:
    """Download to a verified temporary file; never overwrite an existing path."""
    if os.path.lexists(destination):
        verify_file(destination, spec)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{destination.name}-", dir=destination.parent)
    temp = Path(temp)
    try:
        with os.fdopen(fd, "wb") as out, urlopen(url, timeout=60) as response:
            received = 0
            while block := response.read(CHUNK):
                received += len(block)
                require(received <= spec["size"], "Download exceeds locked file size")
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        verify_file(temp, spec)
        try:
            os.link(temp, destination)
        except FileExistsError:
            verify_file(destination, spec)
    finally:
        temp.unlink(missing_ok=True)


def bootstrap(lock: dict, assets: Path) -> None:
    source = lock["source"]
    for name, spec in source["files"].items():
        url = (f"https://raw.githubusercontent.com/{source['repo']}/"
               f"{source['revision']}/{source['directory']}/{leaf(name)}")
        fetch_file(url, assets / name, spec)
    load_edit(lock, assets)


def download(lock: dict, base: Path) -> None:
    require(not base.is_file(), "download --base must be a directory")
    model = lock["model"]
    for name, spec in model["files"].items():
        url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{leaf(name)}"
        fetch_file(url, base / name, spec)
    verify_base(lock, base)


def load_edit(lock: dict, assets: Path) -> tuple[dict, dict]:
    for name, spec in lock["source"]["files"].items():
        verify_file(assets / name, spec)
    manifest = read_json(assets / "manifest.json")
    tensors = manifest["tensors"]
    names = [t["name"] for t in tensors]
    require(names == lock["expected_tensors"], "Unexpected, missing, or reordered edit tensors")
    require(len(set(names)) == len(names) == manifest["changed_tensors"], "Duplicate edit tensors")
    require(all(t["form"] == "sparse" and t["dtype"] == "bfloat16" for t in tensors),
            "Only this published sparse BF16 repair is supported")
    with np.load(assets / "edit.npz", allow_pickle=False) as archive:
        expected = {f"t{i}_{suffix}" for i in range(len(tensors)) for suffix in ("idx", "val")}
        require(set(archive.files) == expected, "Unexpected edit archive arrays")
        arrays = {key: archive[key].copy() for key in archive.files}
    require(sum(t["changed_entries"] for t in tensors) == manifest["changed_entries_total"],
            "Inconsistent manifest edit count")
    return manifest, arrays


def read_header(path: Path) -> tuple[dict, int]:
    """Validate safetensors bounds before reading selected BF16 tensors."""
    size = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(8)
        require(len(prefix) == 8, "Truncated safetensors prefix")
        n = struct.unpack("<Q", prefix)[0]
        require(2 <= n <= min(100_000_000, size - 8), "Invalid safetensors header length")
        header = json.loads(f.read(n), object_pairs_hook=_unique_object)
    require(isinstance(header, dict), "Invalid safetensors header")
    spans = []
    for name, info in header.items():
        if name == "__metadata__":
            require(isinstance(info, dict) and all(isinstance(k, str) and isinstance(v, str)
                    for k, v in info.items()), "Invalid safetensors metadata")
            continue
        require(isinstance(info, dict), f"Invalid tensor entry: {name}")
        shape, dtype, offsets = info["shape"], info["dtype"], info["data_offsets"]
        require(isinstance(shape, list) and all(type(d) is int and d >= 0 for d in shape),
                f"Invalid shape: {name}")
        require(dtype in DTYPE_BYTES, f"Unsupported tensor dtype: {dtype}")
        require(isinstance(offsets, list) and len(offsets) == 2
                and all(type(x) is int for x in offsets), f"Invalid offsets: {name}")
        a, b = offsets
        require(0 <= a <= b <= size - 8 - n, f"Tensor out of file bounds: {name}")
        require(b - a == math.prod(shape) * DTYPE_BYTES[dtype], f"Tensor size mismatch: {name}")
        spans.append((a, b))
    end = 0
    for a, b in sorted(spans):
        require(a == end, "Safetensors ranges overlap or leave a gap")
        end = b
    require(end == size - 8 - n, "Unindexed safetensors trailing data")
    return header, 8 + n


def base_directory(base: Path, lock: dict) -> Path:
    if base.is_file():
        require(base.name in lock["model"]["files"] and base.name.endswith(".safetensors"),
                "Explicit base file must be a locked safetensors shard")
        return base.parent
    return base


def verify_base(lock: dict, base: Path) -> tuple[Path, dict]:
    base = base_directory(base, lock)
    for name, spec in lock["model"]["files"].items():
        verify_file(base / name, spec)
    index = read_json(base / "model.safetensors.index.json")["weight_map"]
    require(isinstance(index, dict) and bool(index), "Empty checkpoint index")
    shards = {leaf(v) for v in index.values()}
    locked_shards = {n for n in lock["model"]["files"] if n.endswith(".safetensors")}
    require(shards == locked_shards, "Index differs from locked shard set")
    for name in shards:
        header, _ = read_header(base / name)
        actual = {k for k in header if k != "__metadata__"}
        require(actual == {k for k, v in index.items() if v == name}, "Index/header mismatch")
    return base, index


def replacement(path: Path, tensor: dict, arrays: dict, i: int) -> tuple[int, bytes, dict]:
    header, start = read_header(path)
    name = tensor["name"]
    require(name in header, f"Missing tensor: {name}")
    info = header[name]
    require(info["dtype"] == "BF16" and info["shape"] == tensor["shape"],
            f"Wrong dtype/shape: {name}")
    a, b = info["data_offsets"]
    with path.open("rb") as f:
        f.seek(start + a)
        raw = f.read(b - a)
    require(hashlib.sha256(raw).hexdigest() == tensor["base_sha256"], f"Base tensor mismatch: {name}")
    values = np.frombuffer(raw, dtype="<u2").copy()
    idx, val = arrays[f"t{i}_idx"], arrays[f"t{i}_val"]
    require(idx.ndim == val.ndim == 1 and idx.dtype.kind in "iu" and val.dtype == np.dtype("uint16"),
            f"Invalid sparse array types: {name}")
    require(len(idx) == len(val) == tensor["changed_entries"] and len(idx) > 0,
            f"Invalid sparse array lengths: {name}")
    require(int(idx.min()) >= 0 and int(idx.max()) < len(values), f"Sparse index out of range: {name}")
    require(len(np.unique(idx)) == len(idx), f"Duplicate sparse indices: {name}")
    changed = int(np.count_nonzero(values[idx] != val))
    require(changed == tensor["changed_entries"], f"Changed entry count mismatch: {name}")
    values[idx] = val
    result = values.tobytes()
    digest = hashlib.sha256(result).hexdigest()
    require(digest == tensor["edited_sha256"], f"Edited tensor mismatch: {name}")
    return start + a, result, {"name": name, "base_sha256": tensor["base_sha256"],
                             "edited_sha256": digest, "changed_entries": changed}


def digest_regions(path: Path, excluded: list[tuple[int, int]]) -> str:
    """Hash every byte outside the edited tensor spans, including the full header."""
    h = hashlib.sha256()
    size = path.stat().st_size
    position = 0
    with path.open("rb") as f:
        for a, b in [*sorted(excluded), (size, size)]:
            require(position <= a <= b <= size, "Overlapping/invalid edit spans")
            f.seek(position)
            remaining = a - position
            while remaining:
                block = f.read(min(CHUNK, remaining))
                require(bool(block), "Unexpected EOF")
                h.update(block)
                remaining -= len(block)
            position = b
    return h.hexdigest()


def verify(lock: dict, base: Path, assets: Path) -> dict:
    manifest, arrays = load_edit(lock, assets)
    base, index = verify_base(lock, base)
    results = [replacement(base / index[t["name"]], t, arrays, i)[2]
               for i, t in enumerate(manifest["tensors"])]
    return {"status": "verified", "base_revision": lock["model"]["revision"],
            "source_revision": lock["source"]["revision"], "tensors": results,
            "changed_entries_total": sum(t["changed_entries"] for t in results)}


def publish_directory(staged: Path, output: Path) -> None:
    """Linux atomic rename with RENAME_NOREPLACE: even empty directories are protected."""
    require(sys.platform.startswith("linux"), "Atomic publication requires Linux renameat2")
    libc = ctypes.CDLL(None, use_errno=True)
    require(hasattr(libc, "renameat2"), "renameat2 is unavailable; refusing unsafe fallback")
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staged), -100, os.fsencode(output), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def rebuild(lock: dict, base: Path, assets: Path, output: Path) -> dict:
    # No upstream subprocess: all checks remain active with Python -O/-OO.
    manifest, arrays = load_edit(lock, assets)
    base, index = verify_base(lock, base)
    output = output.absolute()
    require(not os.path.lexists(output), f"Refusing existing output: {output}")
    require(output.suffix != ".safetensors", "--output is a checkpoint DIRECTORY, not a tensor file")
    resolved = output.resolve()
    require(not resolved.is_relative_to(base.resolve()) and not base.resolve().is_relative_to(resolved),
            "Output cannot overlap the immutable base directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.parent.resolve() / output.name
    needed = sum(f["size"] for f in lock["model"]["files"].values()) + 16 * 1024 * 1024
    require(shutil.disk_usage(output.parent).free >= needed, f"Need at least {needed} bytes free")
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        for name in lock["model"]["files"]:
            shutil.copyfile(base / name, staged / name)
        verify_base(lock, staged)
        by_shard = {}
        results = []
        for i, t in enumerate(manifest["tensors"]):
            shard = index[t["name"]]
            offset, data, proof = replacement(staged / shard, t, arrays, i)
            by_shard.setdefault(shard, []).append((offset, data, proof))
            results.append(proof)
        preservation = {}
        for shard, edits in by_shard.items():
            path = staged / shard
            spans = [(offset, offset + len(data)) for offset, data, _ in edits]
            before = digest_regions(path, spans)
            with path.open("r+b") as f:
                for offset, data, _ in edits:
                    f.seek(offset)
                    f.write(data)
                f.flush()
                os.fsync(f.fileno())
            require(digest_regions(path, spans) == before, "Unedited bytes changed")
            with path.open("rb") as f:
                for offset, data, proof in edits:
                    f.seek(offset)
                    require(hashlib.sha256(f.read(len(data))).hexdigest() == proof["edited_sha256"],
                            "Written tensor verification failed")
            preservation[shard] = before
        file_hashes = {name: sha256(staged / name) for name in lock["model"]["files"]}
        report = {"schema_version": 1, "status": "reconstructed",
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "base_repo": lock["model"]["repo"], "base_revision": lock["model"]["revision"],
                  "source_revision": lock["source"]["revision"],
                  "lock_sha256": hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest(),
                  "tensor_checks": results, "changed_entries_total": sum(t["changed_entries"] for t in results),
                  "unedited_region_sha256": preservation, "output_file_sha256": file_hashes,
                  "validation": "CPU reconstruction only; no generation or benchmark claim"}
        (staged / "rebuild.json").write_text(json.dumps(report, indent=2) + "\n")
        for path in staged.iterdir():
            with path.open("rb") as f:
                os.fsync(f.fileno())
        sync_directory(staged)
        publish_directory(staged, output)
        sync_directory(output.parent)
        return report
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def verify_output(lock: dict, base: Path, assets: Path, output: Path) -> dict:
    manifest, arrays = load_edit(lock, assets)
    base, index = verify_base(lock, base)
    record_path = output / "rebuild.json"
    require(record_path.is_file() and not record_path.is_symlink(), "Missing regular reconstruction record")
    record = read_json(record_path)
    expected_identity = {"schema_version": 1, "status": "reconstructed",
                         "base_repo": lock["model"]["repo"], "base_revision": lock["model"]["revision"],
                         "source_revision": lock["source"]["revision"],
                         "lock_sha256": hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()}
    for key, value in expected_identity.items():
        require(record.get(key) == value, f"Reconstruction provenance mismatch: {key}")
    grouped, tensor_checks, file_hashes, preservation = {}, [], {}, {}
    for i, tensor in enumerate(manifest["tensors"]):
        shard = index[tensor["name"]]
        offset, data, proof = replacement(base / shard, tensor, arrays, i)
        grouped.setdefault(shard, []).append((offset, data, proof))
        tensor_checks.append(proof)
    for name, spec in lock["model"]["files"].items():
        path = output / name
        if name not in grouped:
            verify_file(path, spec)
            file_hashes[name] = spec["sha256"]
            continue
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == spec["size"],
                f"Missing/invalid output shard: {name}")
        spans = [(o, o + len(d)) for o, d, _ in grouped[name]]
        preservation[name] = digest_regions(base / name, spans)
        require(preservation[name] == digest_regions(path, spans),
                f"Unedited output bytes differ: {name}")
        with path.open("rb") as f:
            for offset, data, proof in grouped[name]:
                f.seek(offset)
                require(hashlib.sha256(f.read(len(data))).hexdigest() == proof["edited_sha256"],
                        f"Edited output tensor differs: {proof['name']}")
        file_hashes[name] = sha256(path)
    expected_checks = {"tensor_checks": tensor_checks,
                       "changed_entries_total": sum(t["changed_entries"] for t in tensor_checks),
                       "unedited_region_sha256": preservation, "output_file_sha256": file_hashes}
    for key, value in expected_checks.items():
        require(record.get(key) == value, f"Reconstruction provenance mismatch: {key}")
    return {"status": "output_verified", "output": str(output),
            "edited_tensors": len(manifest["tensors"]), "all_other_bytes_preserved": True,
            "provenance_verified": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bootstrap", "download", "preflight", "verify", "rebuild", "verify-output"))
    parser.add_argument("--base", type=Path, default=ROOT / "models/base")
    parser.add_argument("--assets", type=Path, default=ROOT / "assets/p50c")
    parser.add_argument("--output", type=Path, default=ROOT / "models/rebuilt")
    args = parser.parse_args(argv)
    try:
        lock = load_lock(ROOT / "reproduction.lock.json")
        if args.action == "bootstrap":
            bootstrap(lock, args.assets)
            result = {"status": "edit_assets_verified", "assets": str(args.assets)}
        elif args.action == "download":
            download(lock, args.base)
            result = {"status": "base_download_verified", "base": str(args.base)}
        elif args.action == "rebuild":
            result = rebuild(lock, args.base, args.assets, args.output)
        elif args.action == "verify-output":
            result = verify_output(lock, args.base, args.assets, args.output)
        else:
            result = verify(lock, args.base, args.assets)
        print(json.dumps(result, indent=2))
        return 0
    except (ValidationError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

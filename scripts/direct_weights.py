#!/usr/bin/env python3
"""CPU-only sparse BF16 contract for the official Qwen3.5-4B parent.

This module authenticates numeric artifacts. It does not discover edits, execute
producer code, train, load a model, or materialize a checkpoint. Code and the
repository lock are trusted; an adversary rewriting them is outside this boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile

import numpy as np

try:
    from . import pipeline
except ImportError:
    import pipeline

ValidationError = pipeline.ValidationError
require = pipeline.require
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "configs/qwen35-4b-direct.lock.json"
MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SHAPE = [2560, 9216]
LAYERS = 32
FORMAT = "qwen35-4b-sparse-bf16-v1"
SHARDS = {f"model.safetensors-{i:05}-of-00002.safetensors" for i in (1, 2)}
REQUIRED_FILES = SHARDS | {
    "LICENSE", "chat_template.jinja", "config.json", "merges.txt",
    "model.safetensors.index.json", "preprocessor_config.json", "tokenizer.json",
    "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json",
}
# Conservative engineering ceilings, not a claim of causally safe edit strength.
CEILINGS = {
    "max_tensors": 4, "max_selected_entries_total": 65536,
    "max_selected_entries_per_tensor": 65536, "max_selected_fraction": 0.01,
    "max_absolute_delta": 0.125, "max_relative_l2_delta": 0.05,
}
MAX_JSON_BYTES = 32 * 1024 * 1024
PROVENANCE_KEYS = {
    "hypothesis", "failure_record_sha256", "practice_manifest_sha256",
    "localization_record_sha256", "control_protocol_sha256", "producer_code_sha256",
    "forward_observation_sha256",
}
TENSOR_KEYS = {
    "name", "shape", "dtype", "before_sha256", "after_sha256", "indices",
    "before_bits", "after_bits", "selected_entries", "changed_entries",
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(value, label="SHA-256") -> None:
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), f"Invalid {label}")


def _keys(value, expected, label) -> None:
    require(type(value) is dict and set(value) == set(expected), f"Unexpected {label} fields")


def _shape(value) -> bool:
    return type(value) is list and all(type(n) is int for n in value) and value == SHAPE


def _nonfinite(value):
    raise ValidationError(f"Non-finite JSON number: {value}")


def _json_bytes(data: bytes) -> dict:
    require(len(data) <= MAX_JSON_BYTES, "JSON artifact exceeds size limit")
    try:
        result = json.loads(data, object_pairs_hook=pipeline._unique_object,
                            parse_constant=_nonfinite)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError(f"Invalid JSON: {exc}") from exc
    require(type(result) is dict, "Expected JSON object")
    # json.loads can turn 1e999 into infinity without parse_constant being called.
    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), "Non-finite JSON number")
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(result)
    return result


def read_json(path: Path) -> dict:
    with Path(path).open("rb") as source:
        return _json_bytes(source.read(MAX_JSON_BYTES + 1))


def _read_artifact(path: Path) -> bytes:
    with Path(path).open("rb") as source:
        data = source.read(MAX_JSON_BYTES + 1)
    require(len(data) <= MAX_JSON_BYTES, "JSON artifact exceeds size limit")
    return data


def allowed_names() -> set[str]:
    return {f"model.language_model.layers.{i}.mlp.down_proj.weight" for i in range(LAYERS)}


def load_lock(path: Path = DEFAULT_LOCK) -> dict:
    """Read the separate 4B trust anchor; returned _sha256 binds exact lock bytes."""
    raw = _read_artifact(path)
    lock = _json_bytes(raw)
    _keys(lock, {"schema_version", "format", "model", "architecture", "permitted_tensors",
                 "policy_ceilings", "provenance", "trust_note"}, "lock")
    require(type(lock["schema_version"]) is int and lock["schema_version"] == 1
            and lock["format"] == FORMAT, "Unsupported direct lock schema")
    model = lock["model"]
    _keys(model, {"model_id", "revision", "files"}, "model lock")
    require(model["model_id"] == MODEL_ID and model["revision"] == REVISION,
            "Only the pinned official Qwen3.5-4B parent is supported")
    require(type(model["files"]) is dict and set(model["files"]) == REQUIRED_FILES,
            "Lock must authenticate all required shards and sidecars")
    for name, spec in model["files"].items():
        pipeline.leaf(name)
        _keys(spec, {"size", "sha256", "upstream_identity"}, "file lock")
        require(type(spec["size"]) is int and spec["size"] > 0, "Invalid locked size")
        _sha(spec["sha256"])
        oid = spec["upstream_identity"]
        require(isinstance(oid, str) and len(oid) in (40, 64)
                and all(c in "0123456789abcdef" for c in oid), "Invalid upstream identity")
    arch = lock["architecture"]
    require(arch == expected_architecture(), "Wrong 4B architecture in lock")
    tensors = lock["permitted_tensors"]
    require(type(tensors) is dict and set(tensors) == allowed_names(),
            "Lock must enumerate exactly the 32 language MLP down projections")
    for info in tensors.values():
        _keys(info, {"shape", "dtype", "shard"}, "tensor lock")
        require(_shape(info["shape"]) and info["dtype"] == "BF16" and info["shard"] in SHARDS,
                "Wrong 4B tensor shape, dtype or shard")
    require(lock["policy_ceilings"] == CEILINGS, "Unexpected contract policy ceilings")
    lock["_sha256"] = sha_bytes(raw)
    return lock


def expected_architecture() -> dict:
    return {"model_type": "qwen3_5_text", "num_hidden_layers": LAYERS,
            "hidden_size": SHAPE[0], "intermediate_size": SHAPE[1],
            "num_attention_heads": 16, "max_position_embeddings": 262144,
            "dtype": "bfloat16", "layer_types": [
                "full_attention" if i % 4 == 3 else "linear_attention" for i in range(LAYERS)]}


def resolve_readonly(path: Path) -> tuple[Path, list[dict]]:
    """Resolve every component, including model-local and global blob symlinks."""
    pending = list(Path(os.path.abspath(path)).parts[1:])
    resolved, links = Path("/"), []
    traversals = 0
    while pending:
        part = pending.pop(0)
        if part in ("", "."):
            continue
        if part == "..":
            resolved = resolved.parent
            continue
        candidate = resolved / part
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode):
            traversals += 1
            require(traversals <= 64, "Symlink cycle or excessive chain")
            target = os.readlink(candidate)
            links.append({"path": str(candidate), "target": target})
            parts = list(Path(target).parts)
            if os.path.isabs(target):
                resolved, parts = Path("/"), parts[1:]
            pending = parts + pending
        else:
            resolved = candidate
    require(stat.S_ISREG(resolved.stat().st_mode), f"Not a regular baseline file: {path}")
    return resolved, links


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def verify_baseline(lock: dict, base: Path) -> dict:
    """Fully hash locked inputs using read-only handles, then check index/config/headers."""
    base = Path(base).absolute()
    require(base.is_dir(), "Baseline must be a checkpoint directory")
    files, headers = {}, {}
    for name, spec in lock["model"]["files"].items():
        path, links = resolve_readonly(base / name)
        h = hashlib.sha256()
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            require(before.st_size == spec["size"], f"Size mismatch: {name}")
            for block in iter(lambda: source.read(pipeline.CHUNK), b""):
                h.update(block)
            require(_fingerprint(before) == _fingerprint(os.fstat(source.fileno())),
                    f"Baseline changed during authentication: {name}")
        require(h.hexdigest() == spec["sha256"], f"SHA-256 mismatch: {name}")
        require(resolve_readonly(base / name)[0] == path, "Baseline symlink target changed")
        files[name] = {"path": str(path), "symlinks": links, "size": spec["size"],
                       "sha256": h.hexdigest()}
    config = read_json(Path(files["config.json"]["path"]))
    config = config.get("text_config", config)
    require(type(config) is dict, "Invalid text config")
    for key, expected in lock["architecture"].items():
        actual = config.get(key, config.get("torch_dtype") if key == "dtype" else None)
        require(actual == expected, f"Config architecture mismatch: {key}")
    index = read_json(Path(files["model.safetensors.index.json"]["path"]))
    index = index.get("weight_map")
    require(type(index) is dict and bool(index), "Empty checkpoint index")
    require(all(isinstance(k, str) and isinstance(v, str) for k, v in index.items()),
            "Invalid checkpoint index")
    require({pipeline.leaf(v) for v in index.values()} == SHARDS, "Wrong indexed shard set")
    for shard in SHARDS:
        header, start = pipeline.read_header(Path(files[shard]["path"]))
        require(set(header) - {"__metadata__"} == {k for k, v in index.items() if v == shard},
                "Index/header tensor mismatch")
        headers[shard] = (header, start)
    tensors = {}
    for name, expected in lock["permitted_tensors"].items():
        require(index.get(name) == expected["shard"], f"Wrong or missing tensor: {name}")
        header, start = headers[expected["shard"]]
        info = header[name]
        require(info["shape"] == expected["shape"] and info["dtype"] == "BF16",
                f"Wrong tensor shape/dtype: {name}")
        tensors[name] = {**expected, "offset": start + info["data_offsets"][0],
                         "size": info["data_offsets"][1] - info["data_offsets"][0]}
    return {"status": "baseline_verified", "parent": parent_identity(lock),
            "baseline": str(base), "files": files, "tensors": tensors,
            "scope": "Full locked-file SHA-256, config/index/header checks; no model execution"}


def parent_identity(lock: dict) -> dict:
    return {"model_id": MODEL_ID, "revision": REVISION, "lock_sha256": lock["_sha256"]}


def _limits(limits) -> None:
    _keys(limits, CEILINGS, "limit")
    for key, ceiling in CEILINGS.items():
        value = limits[key]
        if type(ceiling) is int:
            require(type(value) is int, f"Expected integer limit: {key}")
        else:
            require(type(value) in (float, int) and math.isfinite(value), f"Invalid limit: {key}")
        require(0 <= value <= ceiling, f"Limit exceeds policy ceiling: {key}")


def _tensor_structure(tensor: dict, lock: dict, limits: dict) -> None:
    _keys(tensor, TENSOR_KEYS, "tensor patch")
    name = tensor["name"]
    require(isinstance(name, str) and name in lock["permitted_tensors"],
            f"Forbidden or undeclared tensor: {name}")
    require(_shape(tensor["shape"]) and tensor["dtype"] == "BF16",
            "Patch tensor must have exact 4B shape and BF16 dtype")
    _sha(tensor["before_sha256"]); _sha(tensor["after_sha256"])
    indices, old, new = (tensor[k] for k in ("indices", "before_bits", "after_bits"))
    require(type(indices) is list and type(old) is list and type(new) is list,
            "Sparse positions and BF16 words must be JSON arrays")
    n = len(indices)
    require(n <= limits["max_selected_entries_per_tensor"], "Per-tensor support exceeds limit")
    require(n / math.prod(SHAPE) <= limits["max_selected_fraction"], "Support fraction exceeds limit")
    require(len(old) == len(new) == n, "Sparse array lengths differ")
    require(type(tensor["selected_entries"]) is int and tensor["selected_entries"] == n,
            "Wrong selected entry count")
    require(type(tensor["changed_entries"]) is int and 0 <= tensor["changed_entries"] <= n,
            "Invalid changed entry count")
    previous = -1
    for index, before, after in zip(indices, old, new):
        require(type(index) is int and 0 <= index < math.prod(SHAPE), "Index out of bounds")
        require(index > previous, "Duplicate or unsorted sparse coordinates")
        previous = index
        for bits in (before, after):
            require(type(bits) is int and 0 <= bits <= 65535, "BF16 words must be uint16 integers")
            require(bits & 0x7f80 != 0x7f80, "Non-finite BF16 word")
    require(sum(a != b for a, b in zip(old, new)) == tensor["changed_entries"],
            "Wrong actual BF16 changed entry count")


def validate_patch_structure(lock: dict, patch: dict) -> None:
    _keys(patch, {"schema_version", "format", "parent", "provenance", "limits", "tensors"}, "patch")
    require(type(patch["schema_version"]) is int and patch["schema_version"] == 1
            and patch["format"] == FORMAT, "Unsupported sparse patch format")
    require(patch["parent"] == parent_identity(lock), "Patch does not bind exact immutable parent")
    provenance = patch["provenance"]
    _keys(provenance, PROVENANCE_KEYS, "provenance")
    require(isinstance(provenance["hypothesis"], str) and 0 < len(provenance["hypothesis"]) <= 8192,
            "A bounded mechanism hypothesis is required")
    for name in PROVENANCE_KEYS - {"hypothesis"}:
        _sha(provenance[name], name)
    _limits(patch["limits"])
    tensors = patch["tensors"]
    require(type(tensors) is list and len(tensors) <= patch["limits"]["max_tensors"],
            "Too many tensor patches")
    names = set()
    for tensor in tensors:
        _tensor_structure(tensor, lock, patch["limits"])
        require(tensor["name"] not in names, "Duplicate tensor patch")
        names.add(tensor["name"])
    require(sum(t["selected_entries"] for t in tensors) <= patch["limits"]["max_selected_entries_total"],
            "Total selected support exceeds limit")


def bf16_float(words: np.ndarray) -> np.ndarray:
    """Decode raw BF16 words exactly; no conversion/rounding of replacement data."""
    return (words.astype(np.uint32) << 16).view(np.float32)


def apply_tensor_patch(original_words: np.ndarray, tensor: dict, lock: dict,
                       limits: dict) -> tuple[np.ndarray, dict]:
    """Validate then replace words in a PRIVATE copy; caller's array is never mutated.

    A trusted forward worker must also validate the complete patch and authenticate
    its loaded checkpoint. This helper alone cannot authenticate other model tensors.
    """
    _limits(limits)
    _tensor_structure(tensor, lock, limits)
    require(isinstance(original_words, np.ndarray) and original_words.dtype == np.dtype("<u2")
            and list(original_words.shape) == SHAPE and original_words.flags.c_contiguous,
            "Input must be contiguous little-endian uint16 words with the exact 4B shape")
    before_hash = hashlib.sha256(memoryview(original_words).cast("B")).hexdigest()
    require(before_hash == tensor["before_sha256"], "Before tensor SHA-256 mismatch")
    positions = np.asarray(tensor["indices"], dtype=np.int64)
    flat = original_words.reshape(-1)
    require(np.array_equal(flat[positions], np.asarray(tensor["before_bits"], dtype="<u2")),
            "Selected before bits differ from authenticated parent")
    result = np.array(original_words, copy=True, order="C", subok=False)
    result.reshape(-1)[positions] = np.asarray(tensor["after_bits"], dtype="<u2")
    after_hash = hashlib.sha256(memoryview(result).cast("B")).hexdigest()
    require(after_hash == tensor["after_sha256"], "After tensor SHA-256 mismatch")
    original_squared = delta_squared = maximum = 0.0
    # Bounded FP64 reductions avoid allocating a full FP32/FP64 model tensor.
    for start in range(0, flat.size, 131072):
        a = bf16_float(flat[start:start + 131072]).astype(np.float64)
        b = bf16_float(result.reshape(-1)[start:start + 131072]).astype(np.float64)
        require(bool(np.isfinite(a).all()) and bool(np.isfinite(b).all()), "Non-finite tensor value")
        delta = b - a
        original_squared += float(np.dot(a, a))
        delta_squared += float(np.dot(delta, delta))
        maximum = max(maximum, float(np.max(np.abs(delta), initial=0)))
    norm, delta_norm = math.sqrt(original_squared), math.sqrt(delta_squared)
    relative = delta_norm / norm if norm else (0.0 if delta_norm == 0 else math.inf)
    require(maximum <= limits["max_absolute_delta"], "Absolute delta exceeds limit")
    require(relative <= limits["max_relative_l2_delta"], "Relative L2 delta exceeds limit")
    changed = int(np.count_nonzero(result.reshape(-1)[positions] != flat[positions]))
    require(changed == tensor["changed_entries"], "Actual changed entry count mismatch")
    return result, {"name": tensor["name"], "shape": SHAPE, "dtype": "BF16",
                    "before_sha256": before_hash, "after_sha256": after_hash,
                    "selected_entries": len(positions), "changed_entries": changed,
                    "selected_fraction": len(positions) / flat.size,
                    "original_l2": norm, "delta_l2": delta_norm,
                    "relative_l2_delta": relative, "max_absolute_delta": maximum}


def validate_patch(lock: dict, base: Path, patch_path: Path) -> dict:
    """Authenticate full parent; validate one tensor at a time; write no model bytes."""
    raw = _read_artifact(patch_path)
    patch = _json_bytes(raw)
    validate_patch_structure(lock, patch)
    baseline = verify_baseline(lock, base)
    results = []
    for tensor in patch["tensors"]:
        info = baseline["tensors"][tensor["name"]]
        path = Path(baseline["files"][info["shard"]]["path"])
        with path.open("rb") as source:
            source.seek(info["offset"])
            data = source.read(info["size"])
        require(len(data) == info["size"], "Truncated tensor read")
        original = np.frombuffer(data, dtype="<u2").reshape(SHAPE)
        edited, report = apply_tensor_patch(original, tensor, lock, patch["limits"])
        results.append(report)
        del edited, original, data
    return {"status": "patch_verified_cpu_only", "parent": parent_identity(lock),
            "patch_sha256": sha_bytes(raw), "tensors": results,
            "selected_entries_total": sum(t["selected_entries"] for t in results),
            "changed_entries_total": sum(t["changed_entries"] for t in results),
            "baseline_files": baseline["files"], "quality_evaluation": "NOT_RUN",
            "scope": "Numeric/integrity contract only; provenance hashes are declarations, not causal proof"}


def publish_json(document: dict, output: Path) -> str:
    """Publish complete numeric JSON via atomic no-replace hardlink, never a baseline.

    The private staging file is preserved on success/failure for audit. A crash
    before link leaves only staging; after link the destination is already complete.
    """
    encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _json_bytes(encoded)
    output = Path(output).absolute()
    require(output.parent.is_dir(), "Publication parent directory must already exist")
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.building-", dir=output.parent)
    with os.fdopen(fd, "wb") as destination:
        destination.write(encoded)
        destination.flush()
        os.fsync(destination.fileno())
    os.link(temporary, output, follow_symlinks=False)
    pipeline.sync_directory(output.parent)
    return sha_bytes(encoded)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("verify-baseline", "inspect-tensors", "validate-patch"):
        child = commands.add_parser(command)
        child.add_argument("--base", type=Path, required=True)
        child.add_argument("--report", type=Path, help="Atomically publish a new report; never overwrite")
        if command == "validate-patch":
            child.add_argument("--patch", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        lock = load_lock(args.lock)
        if args.command == "validate-patch":
            report = validate_patch(lock, args.base, args.patch)
        else:
            report = verify_baseline(lock, args.base)
            if args.command == "inspect-tensors":
                report = {k: v for k, v in report.items() if k != "files"}
        if args.report:
            publish_json(report, args.report)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValidationError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"direct_weights: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

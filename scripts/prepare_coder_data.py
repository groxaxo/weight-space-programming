#!/usr/bin/env python3
"""Prepare complete, deduplicated, task-disjoint train/validation conversations."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

from coder_common import check, digest, fingerprint, loads, normalize_record, write_json
from pipeline import publish_directory


def prepare(inputs, sources, output, seed=20261106, validation_fraction=0.1):
    check(0 < validation_fraction < 0.5, "validation_fraction must be between 0 and 0.5")
    check(len(inputs) == len(sources), "One source identity is required per input")
    output = Path(output).absolute()
    check(not output.exists() and not output.is_symlink(), "Refusing existing dataset output")
    records, rejected, source_proofs = [], Counter(), []
    seen_conversations, seen_prompts = set(), set()
    for path, source in zip(inputs, sources):
        check(set(("name", "revision", "license")).issubset(source), "Source name/revision/license required")
        check(all(isinstance(source[k], str) and source[k] for k in ("name", "revision", "license")), "Empty source identity")
        check(source.get("split", "train") == "train", "Evaluation sources cannot be used for training")
        path = Path(path)
        check(path.is_file() and not path.is_symlink(), "Input must be a regular JSONL file")
        source_proofs.append({**source, "input_sha256": digest(path)})
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            # Invalid records fail closed with a line number; do not silently train on a partial import.
            try:
                raw = loads(line)
                check(raw.get("split", "train") == "train", "Evaluation record in training input")
                record = normalize_record(raw)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            key = fingerprint({"messages": record["messages"], "tools": record["tools"]})
            if key in seen_conversations or record["prompt_hash"] in seen_prompts:
                rejected["duplicate_prompt_or_conversation"] += 1
                continue
            seen_conversations.add(key)
            seen_prompts.add(record["prompt_hash"])
            record["source"] = source["name"]
            # Same task family from different input shards must stay in one split.
            bucket = int(fingerprint([seed, record["task_id"]])[:16], 16) / 2**64
            record["split"] = "validation" if bucket < validation_fraction else "train"
            records.append(record)
    splits = {name: sorted((r for r in records if r["split"] == name), key=lambda r: (r["task_id"], r["prompt_hash"]))
              for name in ("train", "validation")}
    check(all(splits.values()), "Both train and validation must contain examples")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.preparing-", dir=output.parent))
    try:
        for name, rows in splits.items():
            (stage / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        manifest = {"schema_version": 1, "seed": seed, "validation_fraction": validation_fraction,
                    "sources": source_proofs, "rejected": dict(rejected),
                    "splits": {name: {"count": len(rows), "sha256": digest(stage / f"{name}.jsonl"),
                                      "tasks": sorted({r["task_id"] for r in rows})} for name, rows in splits.items()}}
        write_json(stage / "dataset.json", manifest)
        publish_directory(stage, output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def read_dataset(directory):
    directory = Path(directory)
    manifest = loads((directory / "dataset.json").read_text())
    check(manifest.get("schema_version") == 1, "Unsupported dataset manifest")
    splits = {}
    for name in ("train", "validation"):
        path = directory / f"{name}.jsonl"
        spec = manifest["splits"][name]
        check(digest(path) == spec["sha256"], f"Dataset changed since preparation: {name}")
        rows = [loads(line) for line in path.read_text().splitlines() if line.strip()]
        check(len(rows) == spec["count"], "Dataset count mismatch")
        check(all(r["split"] == name for r in rows), "Dataset split mismatch")
        for row in rows:
            validated = normalize_record(row)
            check(validated["prompt_hash"] == row["prompt_hash"], "Dataset prompt identity mismatch")
        check(sorted({r["task_id"] for r in rows}) == spec["tasks"], "Dataset task identity mismatch")
        splits[name] = rows
    for key in ("task_id", "prompt_hash"):
        check(not ({r[key] for r in splits["train"]} & {r[key] for r in splits["validation"]}),
              f"Train/validation leakage: {key}")
    return manifest, splits


def main():
    from method_policy import blocked_legacy_operation
    blocked_legacy_operation("Legacy supervised-training corpus preparation")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--source", type=Path, action="append", required=True, help="JSON source identity for each --input")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, [loads(p.read_text()) for p in args.source], args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()

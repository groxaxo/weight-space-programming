#!/usr/bin/env python3
"""Import a bounded, pinned public coding/tool corpus; never read private chats.

OpenCodeInstruct's pass scores are publisher-reported filtering signals, not our
own correctness proof. Independent executable evaluation remains mandatory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from coder_common import CoderError, canonical, check, digest, encode_record, load_config, loads, normalize_record, write_json
from prepare_coder_data import prepare

SOURCES = {
    "code": {"name": "nvidia/OpenCodeInstruct", "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
             "license": "CC-BY-4.0", "file": "data/train-00000-of-00050.parquet", "split": "train"},
    "tools": {"name": "NousResearch/hermes-function-calling-v1", "revision": "dae3e1d28cfbcf4b915c04ea1e072030529b4bda",
              "license": "Apache-2.0", "file": "func-calling.json", "split": "train"}}
SYSTEM = "You are a coding assistant. Follow the user's requirements, inspect relevant files, use tools accurately, and report only results supported by tool output."


def authenticate_corpus_file(source, path, info):
    check(info.sha == source["revision"], "Different corpus revision")
    matches = [f for f in info.siblings if f.rfilename == source["file"]]
    check(len(matches) == 1, "Missing corpus file metadata")
    entry = matches[0]
    check(path.is_file() and path.stat().st_size == entry.size, "Corpus file size mismatch")
    sha = digest(path)
    if entry.lfs:
        expected = entry.lfs.sha256
        check(sha == expected, "Corpus LFS hash mismatch")
    else:
        expected = entry.blob_id
        blob = path.read_bytes()
        check(hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest() == expected, "Corpus Git blob mismatch")
    return {"source_file_sha256": sha, "upstream_identity": expected, "size": entry.size}


def convert_hermes(row):
    tools = loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    messages, pending, serial = [{"role": "system", "content": SYSTEM}], [], 0
    for message in row["conversations"]:
        role, value = message["from"], message["value"]
        if role == "system":
            continue  # The native Qwen template supplies its own protocol instructions.
        if role == "human":
            messages.append({"role": "user", "content": value})
        elif role == "gpt":
            calls, content, cursor = [], [], 0
            for match in re.finditer(r"<tool_call>(.*?)</tool_call>", value, re.S):
                content.append(value[cursor:match.start()])
                function = loads(match.group(1))
                call_id = f"call_{serial}"
                serial += 1
                calls.append({"id": call_id, "type": "function", "function": function})
                pending.append((call_id, function["name"]))
                cursor = match.end()
            content.append(value[cursor:])
            result = {"role": "assistant", "content": "".join(content).strip()}
            if calls:
                result["tool_calls"] = calls
            messages.append(result)
        elif role == "tool":
            matches = list(re.finditer(r"<tool_response>(.*?)</tool_response>", value, re.S))
            check(matches, "Missing tool response blocks")
            for match in matches:
                response = loads(match.group(1))
                matching = [p for p in pending if p[1] == response["name"]]
                check(matching, "Tool response has no matching call")
                call_id, _ = matching[0]
                pending.remove(matching[0])
                messages.append({"role": "tool", "tool_call_id": call_id,
                                 "content": canonical(response.get("content", response))})
        else:
            raise CoderError(f"Unexpected source role: {role}")
    check(not pending, "Missing source tool response")
    return {"task_id": "hermes:" + row["id"], "messages": messages, "tools": tools}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/qwen35-4b.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-count", type=int, default=512)
    parser.add_argument("--tool-count", type=int, default=128)
    args = parser.parse_args()
    check(args.code_count >= 1 and args.tool_count >= 1, "Counts must be positive")
    check(not args.output.exists(), "Refusing existing corpus output")
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    cfg = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"], trust_remote_code=False)
    args.output.mkdir(parents=True, exist_ok=False)
    paths, identities, counts = [], [], {}
    for name, count in (("code", args.code_count), ("tools", args.tool_count)):
        source = SOURCES[name]
        path = Path(hf_hub_download(source["name"], source["file"], repo_type="dataset", revision=source["revision"]))
        proof = authenticate_corpus_file(source, path,
            HfApi().dataset_info(source["name"], revision=source["revision"], files_metadata=True))
        rows = pq.read_table(path).to_pylist() if name == "code" else loads_array(path)
        random.Random(cfg["seed"]).shuffle(rows)
        accepted, rejected, seen = [], Counter(), set()
        for row in rows:
            if len(accepted) >= count:
                break
            try:
                if name == "code":
                    check(float(row["average_test_score"]) == 1.0, "publisher_test_failure")
                    statuses = loads(row["tests_execution_status"])
                    check(statuses and all(s == "pass" for s in statuses), "publisher_test_failure")
                    record = {"task_id": "oci:" + row["id"], "messages": [
                        {"role": "system", "content": SYSTEM}, {"role": "user", "content": row["input"]},
                        {"role": "assistant", "content": row["output"]}], "tools": []}
                else:
                    record = convert_hermes(row)
                record = normalize_record(record)
                check(record["prompt_hash"] not in seen, "duplicate_prompt")
                encode_record(tokenizer, record, cfg["max_length"])
                seen.add(record["prompt_hash"])
                accepted.append(record)
            except (ValueError, KeyError, TypeError) as error:
                reason = "oversize" if "exceeds max_length" in str(error) else str(error).split(":")[0][:100]
                rejected[reason] += 1
        check(len(accepted) == count, f"Insufficient eligible {name} examples: {len(accepted)} < {count}")
        target = args.output / f"{name}.jsonl"
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted))
        identity = {**source, **proof, "selection": "seeded shuffle; whole conversations; no truncation"}
        write_json(args.output / f"{name}.source.json", identity)
        paths.append(target)
        identities.append(identity)
        counts[name] = {"accepted": len(accepted), "rejected": dict(rejected)}
    manifest = prepare(paths, identities, args.output / "prepared", cfg["seed"])
    write_json(args.output / "import.json", {"counts": counts, "dataset": manifest})
    print(json.dumps(counts, indent=2), flush=True)


def loads_array(path):
    # Shared loads() intentionally supports arrays as well as objects.
    rows = loads(path.read_text())
    check(isinstance(rows, list), "Expected a source JSON array")
    return rows


if __name__ == "__main__":
    main()

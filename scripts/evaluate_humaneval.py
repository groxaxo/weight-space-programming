#!/usr/bin/env python3
"""Pass@1 on all 164 pinned official HumanEval tasks, with isolated execution.

HumanEval is evaluation-only. Neither canonical solutions nor tests are sent to
the model. A public benchmark cannot rule out pretraining contamination.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from coder_common import check, digest, fingerprint, write_json
from evaluate_coder import Client, extract_code, run_python

DATASET = "openai/openai_humaneval"
REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
FILENAME = "openai_humaneval/test-00000-of-00001.parquet"


def load_humaneval():
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    info = HfApi().dataset_info(DATASET, revision=REVISION, files_metadata=True)
    check(info.sha == REVISION, "Wrong HumanEval revision")
    spec = next(f for f in info.siblings if f.rfilename == FILENAME)
    path = Path(hf_hub_download(DATASET, FILENAME, repo_type="dataset", revision=REVISION))
    check(path.stat().st_size == spec.size, "HumanEval artifact size mismatch")
    if spec.lfs:
        check(digest(path) == spec.lfs.sha256, "HumanEval LFS hash mismatch")
    else:
        import hashlib
        blob = path.read_bytes()
        check(hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest() == spec.blob_id, "HumanEval Git blob mismatch")
    rows = pq.read_table(path).to_pylist()
    check(len(rows) == 164 and len({row["task_id"] for row in rows}) == 164, "Incomplete HumanEval dataset")
    return rows, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18084/v1")
    parser.add_argument("--model", default="wsp-qwen35-4b-coder")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists() and not args.output.is_symlink(), "Refusing existing HumanEval report")
    rows, path = load_humaneval()
    check(run_python("x=1", "assert x==1", args.image)["passed"], "HumanEval sandbox self-test failed")
    client = Client(args.base_url, args.model)
    results = []
    for row in sorted(rows, key=lambda r: int(r["task_id"].split("/")[-1])):
        start = time.monotonic()
        result = {"id": row["task_id"], "passed": False}
        # Only the problem prompt is included. Keep the tests and reference solution out of this request.
        prompt = ("Complete the Python function below. Return complete executable Python code, including the function "
                  "signature and required imports. Use only the standard library. Return one Python code block. "
                  "Do not repeat the problem's docstring or provide explanations.\n\n" + row["prompt"])
        try:
            message, usage = client.complete([{"role": "user", "content": prompt}])
            code = extract_code(message.get("content"))
            result.update(run_python(code, row["test"] + "\ncheck(" + row["entry_point"] + ")", args.image),
                          generated_code=code, usage=usage)
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        result["seconds"] = time.monotonic() - start
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "generated_code"}), flush=True)
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET, "dataset_revision": REVISION, "dataset_file_sha256": digest(path), "license": "MIT",
        "benchmark": "HumanEval pass@1, greedy, 1024 output-token budget", "model": args.model,
        "identity": client.identity, "sandbox_image": args.image, "evaluator_sha256": digest(Path(__file__)),
        "sandbox_runner_sha256": digest(Path(__file__).with_name("evaluate_coder.py")),
        "cases_sha256": fingerprint([{k: row[k] for k in ("task_id", "prompt", "test", "entry_point")} for row in rows]),
        "passed": sum(row["passed"] for row in results), "total": 164, "results": results,
        "scope": "Public HumanEval only; pretraining contamination is not ruled out. Not a frontier-model equivalence claim."}
    write_json(args.output, report)
    print(json.dumps({"passed": report["passed"], "total": report["total"]}), flush=True)


if __name__ == "__main__":
    main()

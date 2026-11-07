#!/usr/bin/env python3
"""Regrade saved HumanEval outputs after a sandbox-only fix; never regenerate answers."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from coder_common import check, digest, fingerprint, loads, write_json
from evaluate_coder import run_python
from evaluate_humaneval import DATASET, REVISION, load_humaneval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-runner-source", type=Path, required=True,
                        help="Frozen evaluate_coder.py matching the original report's runner hash")
    args = parser.parse_args()
    check(not args.output.exists() and not args.output.is_symlink(), "Refusing existing rescoring output")
    report = loads(args.input.read_text())
    check(report["dataset"] == DATASET and report["dataset_revision"] == REVISION, "Unknown generation dataset")
    check(report["evaluator_sha256"] == digest(Path(__file__).with_name("evaluate_humaneval.py")),
          "HumanEval generation implementation changed; rerun generation instead")
    check(digest(args.generation_runner_source) == report["sandbox_runner_sha256"],
          "Frozen generation runner does not match the original report")
    old_tree = ast.parse(args.generation_runner_source.read_text())
    current_runner = Path(__file__).with_name("evaluate_coder.py")
    new_tree = ast.parse(current_runner.read_text())
    for name in ("Client", "extract_code"):
        def node(tree):
            matches = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
            check(len(matches) == 1, "Missing generation or extraction implementation")
            return ast.dump(matches[0], include_attributes=False)
        check(node(old_tree) == node(new_tree), "Generation or code extraction changed; rerun generation instead")
    rows, dataset_path = load_humaneval()
    cases = {r["task_id"]: r for r in rows}
    check(report["dataset_file_sha256"] == digest(dataset_path), "Different dataset bytes")
    check(report["cases_sha256"] == fingerprint([{k: row[k] for k in ("task_id", "prompt", "test", "entry_point")} for row in rows]),
          "Different generation problems or test cases")
    check(report["total"] == 164 and len(report["results"]) == 164 and
          {r["id"] for r in report["results"]} == cases.keys(), "Incomplete original benchmark")
    results = []
    for original in report["results"]:
        row = dict(original)
        row["original_passed"] = row["passed"]
        if isinstance(row.get("generated_code"), str):
            case = cases[row["id"]]
            start = time.monotonic()
            row.update(run_python(row["generated_code"], case["test"] + "\ncheck(" + case["entry_point"] + ")",
                                  report["sandbox_image"]))
            row["rescoring_seconds"] = time.monotonic() - start
        else:
            check(row["passed"] is False, "Passing row is missing its generated code")
        results.append(row)
    result = {**report, "results": results, "passed": sum(r["passed"] for r in results),
        "rescored_at_utc": datetime.now(timezone.utc).isoformat(), "generation_report_sha256": digest(args.input),
        "generation_sandbox_runner_sha256": report["sandbox_runner_sha256"],
        "sandbox_runner_sha256": digest(Path(__file__).with_name("evaluate_coder.py")),
        "rescorer_sha256": digest(Path(__file__)),
        "scoring_note": "Same saved model outputs, completion-checked execution; no answer regeneration"}
    write_json(args.output, result)
    print(json.dumps({"passed": result["passed"], "total": result["total"],
                      "changed_scores": sum(r["passed"] != r["original_passed"] for r in results)}))


if __name__ == "__main__":
    main()

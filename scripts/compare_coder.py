#!/usr/bin/env python3
"""Fail-closed promotion gate for paired, complete baseline/candidate evaluations."""
import argparse
import json
import math
import re
from pathlib import Path
from coder_common import adapter_identity, check, digest, fingerprint, loads, write_json
from evaluate_coder import CODE_CASES, TOOL_CASES


def compare(baseline, candidate):
    for report in (baseline, candidate):
        identity = report['identity']
        check(identity.get('identity_schema') == 2, 'Missing complete serving identity; legacy reports require their frozen gate')
        check(isinstance(identity.get('model_proof_sha256'), str) and re.fullmatch('[a-f0-9]{64}', identity['model_proof_sha256']),
              'Missing authenticated model proof identity')
        check(isinstance(identity.get('limits'), dict) and set(identity['limits']) == {'context', 'output'}, 'Missing serving limits')
        limits = identity['limits']
        check(all(type(v) is int for v in limits.values()) and 0 < limits['output'] < limits['context'] <= 32768,
              'Invalid serving limits')
        if identity.get('adapter'):
            for key in ('adapter_sha256', 'adapter_config_sha256', 'training_result_sha256'):
                check(isinstance(identity.get(key), str) and re.fullmatch('[a-f0-9]{64}', identity[key]),
                      'Missing authenticated adapter identity: ' + key)
    for key in ('model_proof_sha256', 'limits'):
        check(baseline['identity'][key] == candidate['identity'][key], 'Different inference profile: ' + key)
    check(baseline["schema_version"] == candidate["schema_version"] == 1, "Unsupported evaluation schema")
    check(baseline["suite_sha256"] == candidate["suite_sha256"], "Different evaluation suites")
    check(baseline["suite_sha256"] == fingerprint([CODE_CASES, TOOL_CASES]), "Unknown development suite identity")
    check(baseline["evaluator_sha256"] == candidate["evaluator_sha256"] == digest(Path(__file__).with_name("evaluate_coder.py")),
          "Different or unknown evaluation implementation")
    check(baseline["sandbox_image"] == candidate["sandbox_image"], "Different evaluation environments")
    check(baseline["identity"]["base_model"] == candidate["identity"]["base_model"] and
          baseline["identity"]["revision"] == candidate["identity"]["revision"], "Different base model identities")
    check(baseline["identity"].get("quantization") == candidate["identity"].get("quantization"), "Different quantization modes")
    check(baseline["identity"].get("runtime") == candidate["identity"].get("runtime"), "Different inference runtimes")
    check(baseline["identity"]["protocol_sha256"] == candidate["identity"]["protocol_sha256"], "Different serving protocols")
    check(not baseline["identity"]["adapter"] and candidate["identity"]["adapter"], "Need an actual base/candidate pair")
    by_id = []
    for report in (baseline, candidate):
        rows = report["results"]
        check(rows and len({r["id"] for r in rows}) == len(rows), "Missing or duplicate result rows")
        check(all(type(r["passed"]) is bool and r["category"] in ("code", "tools") for r in rows), "Invalid result row")
        by_id.append({r["id"]: r for r in rows})
    first, second = by_id
    check(first.keys() == second.keys(), "Incomplete paired evaluation")
    expected_ids = {row[0] for row in CODE_CASES} | {row["id"] for row in TOOL_CASES}
    check(first.keys() == expected_ids, "Missing required development cases")
    check(all(first[k]["category"] == second[k]["category"] for k in first), "Mismatched result categories")
    regressions = [k for k in first if first[k]["passed"] and not second[k]["passed"]]
    improvements = [k for k in first if not first[k]["passed"] and second[k]["passed"]]
    code = [r for r in second.values() if r["category"] == "code"]
    tools = [r for r in second.values() if r["category"] == "tools"]
    check(code and tools, "Both coding and tool evaluations are required")
    accepted = (not regressions and bool(improvements) and sum(r["passed"] for r in code) / len(code) >= 0.8
                and all(r["passed"] for r in tools))
    return {"status": "development_gate_passed" if accepted else "not_promoted", "regressions": regressions,
            "improvements": improvements, "requires_real_client_acceptance": True,
            "scope": "Passing this gate alone does not authorize deployment or establish broad coding reliability."}


def acceptance(baseline, candidate, baseline_human, candidate_human, clients, training):
    """Require broader paired results and actual client edits before local acceptance."""
    development = compare(baseline, candidate)
    from evaluate_humaneval import DATASET, REVISION
    expected_ids = {f"HumanEval/{i}" for i in range(164)}
    for development_report, human in ((baseline, baseline_human), (candidate, candidate_human)):
        check(human["schema_version"] == 1 and human["dataset"] == DATASET and
              human["dataset_revision"] == REVISION, "Unknown HumanEval dataset")
        check(human["dataset_file_sha256"] == "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6",
              "Unknown HumanEval dataset bytes")
        check(human["evaluator_sha256"] == digest(Path(__file__).with_name("evaluate_humaneval.py")),
              "Unknown HumanEval evaluator")
        check(human["sandbox_runner_sha256"] == digest(Path(__file__).with_name("evaluate_coder.py")),
              "Unknown HumanEval sandbox runner")
        check(human["identity"] == development_report["identity"] and
              human["sandbox_image"] == development_report["sandbox_image"], "HumanEval model or sandbox mismatch")
        rows = human["results"]
        check(len(rows) == 164 and {r["id"] for r in rows} == expected_ids and
              all(type(r["passed"]) is bool for r in rows), "Incomplete HumanEval results")
        check(human["total"] == 164 and human["passed"] == sum(r["passed"] for r in rows),
              "Inconsistent HumanEval totals")
    for key in ("cases_sha256", "benchmark", "evaluator_sha256"):
        check(baseline_human[key] == candidate_human[key], "Different HumanEval evaluations")
    check(clients["identity"] == candidate["identity"] and clients["sandbox_image"] == candidate["sandbox_image"],
          "Client acceptance model or sandbox mismatch")
    check(clients.get("harness_sha256") == digest(Path(__file__).resolve().parents[1] / "ops/smoke_clients.py"),
          "Different or unknown client acceptance harness")
    rows = clients["results"]
    check(len(rows) == 2 and {r["client"] for r in rows} == {"opencode", "hermes"} and
          all(type(r["passed"]) is bool for r in rows), "Missing real client results")
    client_pass = all(r["passed"] and r.get("exit_code") == 0 and not r.get("timed_out", True)
                      and r.get("independent_tests", {}).get("passed") is True
                      and r.get("independent_tests", {}).get("tests_completed") is True for r in rows)
    check(clients["passed"] == all(r["passed"] for r in rows), "Inconsistent client totals")
    check(training["status"] == "trained_unpromoted" and training["smoke_only"] is False,
          "Need a complete training candidate")
    before, after = training["before"]["eval_loss"], training["after"]["eval_loss"]
    check(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
              for v in (before, after)), "Invalid validation loss")
    for key, value in adapter_identity(training).items():
        check(candidate['identity'].get(key) == value, 'Training result belongs to a different adapter: ' + key)
    proxy_hash = digest(Path(__file__).resolve().parents[1] / 'ops/client_proxy.py')
    for row in rows:
        proof = row.get('model_attestation', {})
        check(proof.get('verified') is True and proof.get('requests', 0) > 0 and
              proof.get('identity_sha256') == fingerprint(candidate['identity']) and
              proof.get('proxy_sha256') == proxy_hash and bool(proof.get('instance_id')),
              'Missing actual client request attribution')
    broad_pass = candidate_human["passed"] >= baseline_human["passed"]
    accepted = development["status"] == "development_gate_passed" and broad_pass and client_pass and after < before
    return {"status": "accepted_for_local_use" if accepted else "not_promoted", "development": development,
        "humaneval": {"baseline": baseline_human["passed"], "candidate": candidate_human["passed"], "total": 164,
                      "aggregate_nonregression": broad_pass}, "real_clients_passed": client_pass,
        "validation_loss_improved": after < before,
        "scope": "Measured Python and isolated file-tool tasks only; broader coding reliability remains unproven."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-humaneval", type=Path)
    parser.add_argument("--candidate-humaneval", type=Path)
    parser.add_argument("--clients", type=Path)
    parser.add_argument("--training", type=Path)
    args = parser.parse_args()
    pair = [loads(args.baseline.read_text()), loads(args.candidate.read_text())]
    extra = [args.baseline_humaneval, args.candidate_humaneval, args.clients, args.training]
    check(not any(extra) or all(extra), "All broader acceptance inputs are required together")
    result = acceptance(*pair, *[loads(p.read_text()) for p in extra]) if all(extra) else compare(*pair)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] in ("development_gate_passed", "accepted_for_local_use") else 2)


if __name__ == "__main__":
    main()

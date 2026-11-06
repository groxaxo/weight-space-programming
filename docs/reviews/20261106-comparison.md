# Independent review versus GPT-6.1 Sol Max

## Scope and sequence

Both reviews target initial commit `5367eaa`, which contained just `.gitignore`, `README.md`, and the 57-line `scripts/pipeline.py`. The independent review findings were recorded in `independent-initial-20261106.txt` before reading Sol's completed response.

Sol session `01a1af02-5213-7461-b8c6-62bd19c11e10` finished a **static** review. Its process tools were blocked, so it did not run tests or reconstruct weights. The earlier implementation session also finished without changes because its command and write paths failed. Consequently, the expanded autonomous pipeline described in earlier conversation was not in the repository.

This PR's repairs and executed validation were performed by the independent reviewer, not Sol. Sol has not reviewed the remediation diff. No new Sol or Astra agent run was required for this review.

## Findings and resolution

Original line numbers below refer to `5367eaa:scripts/pipeline.py` unless stated otherwise.

| Issue | Sol's review | Independent assessment | Remediation |
|---|---|---|---|
| Runtime checks use `assert` (24–27, 36–41); upstream hashes do too | P1 | Agree; optimized Python can defeat both integrity and base-overwrite guards | Explicit exceptions; no upstream subprocess; optimized-mode tests |
| Edit archive/manifest/source have no enforced integrity anchor (24–28, 41) | P1 | Agree; merely printing a checksum does not authenticate anything | Trusted Git lock with exact artifact hashes, revisions, ordered tensor set; no upstream code execution |
| Only the two edited base tensors are authenticated (41–42) | P2 | Agree; the hardcoded base revision could falsely describe corrupted untouched weights | Entire base snapshot and every sidecar authenticated against immutable source identities |
| Fresh clone cannot run because `upstream/` is ignored and no bootstrap exists | P2 | Operational blocker for reproducing this repo on another host | Pinned bootstrap/download commands; a real clean-clone bootstrap check is recorded separately |
| Untracked `reports/` directory missing (43) | P2 | Agree; failure occurs after writing a large output and blocks retry | Provenance is created inside the private checkpoint staging directory before publication |
| Partial output on interruption/disk exhaustion (39–43) | P2 | Agree; additionally, existence checks are racy across concurrent writers | Verified staging + Linux no-replace atomic directory publication; actual concurrency tests |
| No regression tests | P2 | Agree; cloned OpenWSP tests do not cover this wrapper | 40 CPU regression tests and real-model validation |
| Default `model.safetensors` does not exist (49) | Absent from final reviewer findings; confirmed in Sol's separate implementation log | P1 common-path failure; renaming the literal is less robust than following the index | Resolve the pinned index/shard set; accept directory or exact shard |
| Output lacks config/tokenizer/index | Not included in final review | An isolated tensor file is not the complete checkpoint the user expects | Preserve and authenticate all model sidecars |
| Autonomous research loop/evaluation/budget controls absent | Review scope was reconstruction wrapper | Critical scope clarification: do not label the whole requested agent system complete | README explicitly enumerates implemented versus absent capabilities |

## Difference in perspective

Sol's strongest contribution was precise identification of the integrity trust gap: a valid hash check is meaningless if the expected manifest can itself be silently replaced. The independent review agrees with all seven final findings, but treats the broken fresh-clone path and incorrect default checkpoint filename as immediate operational blockers, not just documentation improvements.

The implementation fixes the pinned reproduction path instead of introducing an unreviewed general-purpose agent executor. That is deliberate: a verified baseline should exist before adding model-written code, API spending, or benchmark-selection logic. The PR does not claim the planned Sol-default/Astra-escalation system is implemented.

## Validation and residual scope

The CPU suite covers input/archive corruption, missing/reordered tensors, wrong shapes/hashes, unsafe shard paths, disallowed pickle arrays, overwrite attempts, empty-directory and dangling-symlink collisions, competing processes, optimized Python, copy/report failures, low disk, and preservation of untouched tensors/metadata. Real reconstruction used the existing pinned Qwen checkpoint, and bootstrap fetched the published edit files independently from the pinned research commit.

Full model and tensor hashes, executed commands, local test logs, and the clean-clone smoke result are preserved in `validation-20261106.json` and accompanying evidence files. Benchmark evaluation and inference have not been run. No running GPU service was stopped or reconfigured. Model binaries, cached assets, secrets, and raw agent session logs are excluded from Git.

The threat model assumes a trusted checkout/lock and operator-controlled filesystem. The implementation is not an isolation boundary for untrusted autonomous code. Publication requires Linux renameat2; Windows/macOS support is not claimed. Interrupted single-file downloads do not yet resume byte ranges.

# Pipeline review: nine fixes and .54 verification

Date: 7 November 2026, Pacific/Auckland. Original reviewed commit: `d2a92cf3b5fbc4ea4debdfa7a488cf5136a1893e`.

**All nine reviewed discrepancies are fixed.** The source now has 136 regression/control tests, including 34 added for this work. The complete .54 environment passes all 136; the minimal CPU environment passes with five expected optional training-library skips. The companion [machine-readable resolution record](20261107-pipeline-fixes.json) binds the implementation and verification evidence.

The live control used the actual pinned Qwen3.5-4B snapshot, a complete two-step epoch on 16 training/four validation public conversations, checkpoint-1 export, HF inference, installed OpenCode/Hermes clients, CPU compilation, FP32 LoRA export, manifest creation and native inference. This verifies the pipeline stages. Its small adapter remains unpromoted.

| Finding | Resolution | Regression evidence |
|---|---|---|
| F01 (P1) | All checkouts share a host lock and durable owner lease. Cleanup is idempotent and cannot restore an older owner inside a newer GPU window. Abandoned leases recover under the lock; initially inactive services stay inactive. A bounded post-stop readiness wait retains strict GPU admission and recovery uses only the standard library. | [test_gpu_supervisor.py](../../tests/test_gpu_supervisor.py) |
| F02 (P1) | Every candidate requires valid weight, configuration and training-result identities. Complete acceptance also requires attestations for actual client requests. Missing/legacy/mismatched evidence fails before a quality verdict. | [test_pipeline_contracts.py](../../tests/test_pipeline_contracts.py) |
| F03 (P2) | Training captures and authenticates its recipe/configuration. Selection verifies that archive, preserves the parent source, stores selection source separately and atomically publishes a complete export. Failures remove only staging, allowing retry. | [test_artifact_flow.py](../../tests/test_artifact_flow.py) |
| F04 (P2) | Fenced examples and valid Python are literal output. Native function blocks are recognized at top-level call boundaries, with strict argument validation and malformed-call rejection preserved. | [test_pipeline_contracts.py](../../tests/test_pipeline_contracts.py) |
| F05 (P2) | Schema-2 profile identity binds input proof, effective limits, adapter weights/configuration and training-result fingerprint. A separate per-process nonce binds individual JSON/SSE requests without preventing paired runs across processes. | [test_pipeline_contracts.py](../../tests/test_pipeline_contracts.py) |
| F06 (P2) | The authenticated index must name exactly the proof’s weight shards. Every loader file present in the snapshot must be authenticated. Changed or omitted weights/sidecars are rejected. | [test_pipeline_contracts.py](../../tests/test_pipeline_contracts.py) |
| F07 (P2) | Nested tool arguments and tool descriptions/schemas receive the same delimiter and credential-pattern validation as message content before native-template rendering. | [test_pipeline_contracts.py](../../tests/test_pipeline_contracts.py) |
| F08 (P2) | Both installed-client harnesses use a loopback proxy that validates every completion response/chunk against the expected model and server instance, requires actual completion traffic and records an attestation. Unsupported capability probes return 404 and never count as completions. | [test_client_proxy.py](../../tests/test_client_proxy.py) |
| F09 (P2) | The committed producer authenticates inputs, runs CPU Q8 conversion, creates compiled or unmerged manifests, supports authenticated base reuse and publishes atomically. Its output is consumed by the native supervisor without hand-authored fields. | [test_artifact_flow.py](../../tests/test_artifact_flow.py) |

## Actual model and client checks

| Profile | Python tasks | Tool scenarios | OpenCode edit | Hermes edit | Attributed requests |
|---|---:|---:|---|---|---|
| hf-base | 6/8 | 6/6 | Pass | Pass | opencode=4, hermes=4 |
| hf-control | 6/8 | 6/6 | Pass | Pass | opencode=4, hermes=4 |
| native | 6/8 | 5/6 | Timeout; frozen regrade passes | Pass | opencode=4, hermes=4 |

The control adapter is checkpoint 1 from the short verification run, not either fully benchmarked lower-rate candidate from the original experiment. The first checkpoint is a zero-update warm-up control (all 248 B factors are zero). The supplemental real CPU check therefore exports checkpoint 2, requires nonzero merged tensor changes and converts that trained update to Q8. Code-quality outcomes remain separate from artifact integrity and request attribution. Every completed client trial has a verified model/instance attestation; independent tests still determine whether its edit is correct.

The real CPU compilation verified all 248 edited tensors and every unedited byte. The LoRA exporter and native frontend exercised the factor-aware unmerged route. A separate real checkpoint-2 compilation changed 248 tensors, and its compiled-to-Q8 conversion exercised the producer’s merged route and passed native-manifest authentication. Large duplicate derived files may be removed only after their hashes/proofs are retained; the delivery record states their final storage disposition.

The development gate returned `not_promoted` for the short control. The complete acceptance gate still requires fresh paired all-164 HumanEval measurements and successful actual clients; unit controls exercise both valid acceptance and rejected evidence. The bounded verification did not regenerate Qwen’s full HumanEval answers. No adapter was promoted.

## HumanEval transport and grader control

All 164 official reference solutions passed through the current `evaluate_humaneval.py`, HTTP client/server protocol and pinned Docker grader. That provider was explicitly identified as `SYNTHETIC-REFERENCE-CONTROL-NO-QWEN`; it performed no Qwen inference. This establishes grader/transport operation and must not be reported as a model score. Its raw report cannot be paired with a Qwen candidate because model identity differs.

The historical model scores remain 128/164 for the base and 127/164 and 126/164 for the fully benchmarked lower-rate candidates. Their original reports and source snapshots remain unchanged. Schema-1 reports are deliberately refused by the new gate; do not retrofit identity fields or combine them with schema-2 measurements.

## Integration correction found during live validation

The initial live trial was stopped after the new proxy rejected Hermes attribution. An actual managed-CLI synthetic-provider control showed that Hermes was using the correct proxy route, while six unsupported capability-discovery requests had been counted as provider failures. Those probes were GET `/api/v1/models`, `/api/tags`, `/v1/props`, `/props`, `/version`, and POST `/api/show`.

The proxy now returns a normal 404 for unsupported capabilities without forwarding them upstream or counting them as completions. Tests require both that probes plus verified completions can pass and that probes alone cannot attest a result. Changed models/instances, missing response identities and completion errors remain rejected. The repeated actual-CLI control passed two completion calls (one SSE and one ordinary JSON), six discovery requests and zero attribution errors. No global Hermes profile or installed client source was changed.

The corrected live trial used a fresh output directory. Its native handoff then correctly refused an immediate post-stop admission before CUDA teardown and/or NVIDIA’s sampled utilization had settled. A bounded 60-second readiness wait was added without lowering the 10,500 MiB/zero-utilization criteria or stopping unknown jobs. Recovery was also verified under `python -S` without NumPy or training dependencies. The native stage alone was rerun successfully in a new output directory; the completed HF results and both failed attempts remain preserved. The aggregate record identifies each source run, and no prior measurement was rewritten. During the parallel CPU conversion, the native OpenCode independent Docker check hit its 15-second wall limit. Its frozen file subsequently passed the identical grader and limits after the conversion finished, with no model regeneration. The original timeout, source/test/runner hashes and separate regrade are retained; the original client report remains a rejection.

## Lifecycle and reproduction

The shared host lock and owner journal cover the HF and native phases. Restoration is idempotent, old callbacks defer to a live owner, abandoned owners recover before admission, and an initially inactive service stays inactive. CPU regressions cover lost markers, exceptions, overlapping checkouts and callback replay. The live run also restored Bonsai; health readiness waits through the service’s temporary 503 warm-up response.

The actual existing p50c 0.8B reconstruction was rechecked with the current verifier, including provenance, edited tensors and all unedited regions. Model, Moss/Bonsai runtime and canonical memory files were preserved. The coding endpoint is stopped after the verification phases.

Use [the coding workflow](../coding-assistant.md) for fresh model/data acquisition, training, atomic export, the committed native-manifest producer, installed-client acceptance and the bounded `ops/verify_pipeline54.py` command. Use unique output directories. Add `--humaneval` and increase the transient-unit runtime if regenerating complete model benchmarks.

The handoff includes the exact source patch/archive, clean-checkout validation, real model/client reports, reference-provider and managed-Hermes controls, reconstruction and final-host checks, checksums and publication/CI metadata. Dependency caches, private client state, model binaries and the private Bonsai unit snapshot are excluded from Git.

This work validates the listed pipeline stages and closes the reviewed logic gaps. Broad repository reliability, other languages, unrestricted agent environments and frontier-model equivalence still require their own evidence.

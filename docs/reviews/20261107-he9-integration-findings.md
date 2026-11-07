# HE9 integration review and findings

The nine original pipeline discrepancies remain closed in their separate frozen finding-to-test record. This follow-up records seven additional integration failures found while replaying the exact historical HE9 set. Each has a regression control. The latest implementation at `c2b4ed05f1b7fd0ff959f85950d76f7310cdb322` passes all 240 tests on .54 with actual pinned Docker and zero skips (39.121 seconds); GitHub Actions run 42347035658 succeeds.

| Finding | Failure and enforced correction | Regression evidence |
|---|---|---|
| HE9-01 | **Recently closed TCP sockets were mistaken for active listeners.** The native readiness probe now uses SO_REUSEADDR, accepting TIME_WAIT while continuing to refuse an active listener. | `test_native_profile.NativeProfileTests.test_native_port_probe_allows_time_wait_and_rejects_a_live_listener` |
| HE9-02 | **Declared sampling settings and seed did not consistently reach native generation.** Request construction forwards the chosen sampling profile, rejects explicit drift and verifies the actual native generation settings per response. The matched pair was restarted after correction. | `test_gguf_backend.GgufBackendTests.test_declared_instruct_sampling_reaches_the_native_engine`, `test_sampling.SamplingTests.test_engine_report_must_confirm_all_effective_settings` |
| HE9-03 | **A 200-message API limit rejected valid conversations far below the token limit.** Conversation validation now follows configured context while preserving the 2 MiB body limit and exact rendered-token/output reservation checks. A real 202-message Qwen request returned READY. | `test_coder.ProtocolTests.test_short_tool_conversations_are_not_rejected_at_200_messages` |
| HE9-04 | **Back-to-back GPU samples could disagree during authorized Bonsai teardown.** Native admission retries the same strict zero-utilization/10500-MiB-free check for at most 60 seconds. Persistent occupancy and unknown layouts remain rejected; unknown jobs are not stopped. | `test_gpu_supervisor.AdmissionTests.test_native_admission_waits_for_a_later_strictly_idle_sample`, `test_gpu_supervisor.RestorationTests.test_readiness_timeout_does_not_relax_memory_or_utilization_requirements` |
| HE9-05 | **The detector missed a real module-before-options unittest command.** Both python3 -m unittest -v test_scheduler and python3 -m unittest test_scheduler -v are recognized, including .py and protected workspace/stderr forms. A frozen-log audit proves old=False/new=True. The original timeout/failing-code verdict stays rejected; both compatibility arms were repeated in fresh directories. | `test_project_command_forms.ProjectCommandForms.test_module_before_verbose_executes_the_same_protected_suite` |
| HE9-06 | **The qualifier assumed the native health shape for the frontend response.** The qualifier now checks the actual single-model discovery body, model identity and nonempty process nonce. A real HTTP-handler RED control reproduced the error; substituted/duplicate/status-only records remain rejected. The corrected complete real pair returns not_promoted for model-quality failures. | `test_he9_qualification.QualificationTests.test_real_frontend_health_discovery_is_accepted_with_matching_client_instances`, `test_he9_qualification.QualificationTests.test_discovery_cannot_omit_or_substitute_the_model_or_instance` |
| HE9-07 | **Client deadline finalization queried discovery before the pending completion drained.** Both harnesses now drain before attribution; only own labeled project containers stop before file hashing/grading. The original incomplete timeout is preserved, and the continuation uses a fresh paired compatibility baseline. | `test_client_finalization.ClientFinalizationTests.test_file_client_timeout_drains_and_attests_without_becoming_a_pass`, `test_client_finalization.ClientFinalizationTests.test_project_client_timeout_drains_and_attests_without_becoming_a_pass` |

## Evidence and quality decisions

The primary base/MLP pair is complete at frozen inference/actor source446c280, and both supplementary client arms are complete at source d143156. Later qualification corrections authenticate those actual ancestor blobs and preserve the reports. All eighteen primary tasks are scored; the selected Base+Plus score improves3/9 to 5/9. The MLP adapter still fails the complete gate: development success drops12/14 to 11/14 and Hermes's two-module project fails. The adapter remains unpromoted.

The original failed readiness controller stopped before any additional training began. Its error and the original compatibility01 interruption are preserved. Fresh resume controllers ran the unchanged predeclared flow recipe at verified commitdb1edf4; its completed training, native export and all-nine model tests remain bound to their actual sources. Its invalid complete compatibility record is retained.

Observed command restrictions are part of the actual client environment. The base Hermes project attempted python -c diagnostics that its single-query approval policy blocked; both saved base and MLP projects also fail independent assertions. The new authored flow trajectories use the permitted unittest path. No approval policy was disabled for the benchmark.

Other boundaries remain explicit: synthetic CLI/provider and reference-grader controls are not Qwen scores; OpenCode step-limit controls did not establish enforcement, so only the archived actor deadline is relied on; thinking-profile malformed tool output stays rejected; max-context allocation does not prove full-window recall; task 32's canonical-root comparison is not official EvalPlus parity. Actual public-example residual audits are labeled separately and never rewrite the frozen verdict.

The complete experiment report and machine-readable findings include all nine tasks for every candidate, every actual client result, training/export identities, qualification and source/resource proof. The final publication manifest records immutable Hub and merged Git commits. Hidden tests, canonical reference implementations, private keys, personal profiles and model binaries stay outside the Git report.

## Continuation capability and controls

A new explicit adapter-initialization path authenticates a completed non-smoke parent, its recipe, every adapter file, pinned base, rank, scale, scope and target set. It then verifies exact FP32 equality of the loaded A/B tensors before the zero-rate memory probe. It starts a new optimizer and records that distinction. Real CPU PEFT controls verify identical restored logits and continued training; negative controls reject altered recipes/files, incomplete runs, mismatched identities/targets and incomplete tensor sets. A supervisor integration control checks that the parent is actually forwarded and is refused for inference phases.

Additional authored data controls require exact string edits to match the observed file, preserve valid Python and correct full function text, retain the actual observed failure/error counts, issue only one check for run-once requests, and keep every previous/new client and repair state of a toy family in the same split. These controls do not establish Qwen competence. The continuation is separately measured through all historical and real-client fixtures; implementation controls alone do not qualify it.

## HE9-07: finalization after a client deadline

A real flow OpenCode project timeout exposed another transport/lifecycle discrepancy: the actor stopped while its final provider completion was still draining, and immediate model discovery timed out. The original row lacks attribution and independent grading, so its complete qualifier correctly fails. Both real-HTTP regression controls reproduce that rejection on the old file/project harnesses. The correction drains pending requests before attribution, preserves the failed timeout verdict and stops only own uniquely labeled project containers before source hashing and grading. It also repeats scoped cleanup on error paths. No approval policy or actor budget is changed.

Controls: `test_client_finalization.ClientFinalizationTests.test_file_client_timeout_drains_and_attests_without_becoming_a_pass` and `test_client_finalization.ClientFinalizationTests.test_project_client_timeout_drains_and_attests_without_becoming_a_pass`. The original incomplete flow record and controller failure remain archived. The continuation uses a fresh complete baseline compatibility arm under the corrected source.

## Model-quality findings and community release

The best completed adapter remains the earlier MLP: 7/9 Base and 5/9 Base+Plus
versus the matched base 6/9 and 3/9, but development drops 12/14 → 11/14 and
Hermes project checks fail. The flow adapter scores 6/9 and 4/9 while losing
base-solved task 76; its original complete client proof is invalid. The
authenticated MLP continuation scores 6/9 and 4/9, with development 9/14.
Its task-101 edit is valid Python but uses `s.split()` and ignores commas.
Its timed-out OpenCode scheduler duplicates normalization, omits unknown-dependency
validation and uses sorted/append-only ready jobs rather than the required
original-input-priority choice. These are model logic failures, preserved under
the fixed deadline and independent assertions.

The selected MLP is published as an experimental community artifact with source,
method, failures, unchanged measured weights, portable config and attribution.
Its complete `not_promoted` decision is retained; no default provider is changed.
The portable PEFT control uses an FP32 shape-only CPU base and verifies exact
loading of all 192 factors, without inference. PEFT can otherwise round factors
through a BF16 base layer; Python loading examples are separate from the measured
Q8/F32 native profile.

Further work should improve family-disjoint executable coverage of chunk
boundaries, half-open intervals, truthful failure observations and cross-module
planning with permitted commands. Repeat seeded paired trials and evaluate unseen
repository tasks before promotion. The later lower-loss runs are evidence that
longer training or validation loss alone is insufficient. No further tuning was
started after the user's closeout instruction.

## Open follow-up: effective Hermes output budget

All 18 malformed final continuation tool calls stop at exactly 1,024 generated
tokens with native `stop_type=limit`. Their saved requests omit both
`max_tokens` and `max_completion_tokens`, so the frontend validator's 1,024-token
default applies even though the server advertises an 8,192-token output cap.
The patch calls end before their closing markup. The parser correctly rejects
these incomplete calls; this is not proof of provider drift or a parser bug.
Hermes reaches its 1,000-second deadline; its original row lacks complete
attribution and independent grading, and the complete qualifier rejects it.
The successful request-drain correction is confirmed by the fully attributed
OpenCode timeout and does not make malformed responses valid.

This is an **open budget/configuration follow-up**, not an eighth fixed
integration discrepancy or a new measured model gain. Explicitly configure and
record a larger output budget and repeat both compatibility arms before drawing
a training-only explanation for this failure. No such new run is claimed here.
A separate diagnostic of the stopped Hermes workspace also fails independent
Docker assertions, without altering the original row or its missing proof.
The saved-request and diagnostic evidence is retained in the report JSON.

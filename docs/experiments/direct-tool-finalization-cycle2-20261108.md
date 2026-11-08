# Qwen3.5-4B direct cycle 2: tool-action completion

The experiment ended **`STOP_semantic_specificity`**, execution complete. B and P
matched their required behavior; S, C and N did not. In particular, the model
also repeats the parser call after a successful result, and completion/neutral
cues produce identical fenced JSON. The proposed nonzero-result-specific
mechanism is falsified. **No weights were edited, no candidate was created, and
no improvement over the base has been demonstrated.**

## Objective and origin

The user requested a new direct-weight-editing hypothesis on `.54`, preserving
artifacts and unrelated jobs, with independent paired verification and a push
to `main`. The method excludes LoRA/QLoRA, PEFT training, gradients, optimizers,
loss fitting and trained-adapter merging. The original `0be1a86` project
reconstructs a published 0.8B patch; it supplies no general discovery algorithm
or 4B coordinates. The [pinned upstream exp8 rules](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md)
remain authoritative for this method.

The earlier interval-boundary study stays closed as `STOP_controls_failed`.
Its reverse joint margin remained positive (+0.8176). This cycle uses none of
its layer groups, features, donors or proposed support subdivisions.

The new exposed development target came from a historical native Q8 base-model
failure: after a completed `run_tests(target="parser")` result with exit code 1,
four passes and two failures, the model called the same tool again instead of
returning the requested result fields. The hypothesis predicts that a sparse
MLP write wrongly retains an outstanding-action state after a failed result.
It predicts normal termination after a success or an explicit completion cue,
persistence with a length-matched neutral cue, and a legitimate second call
when the instruction actually requires two runs. Each is a required prediction.

The user-selected surgeon was launched as Codex `gpt-6.1-sol`, reasoning `xhigh`,
with a fresh context, task `/root/direct_tool_surgeon`. The independent custodian
was separately launched as Codex `gpt-6.1-sol`, reasoning `max`, task
`/root/independent_custodian`, after the user's explicit selection. These are
recorded invocation settings; provider-side model attestation is unavailable.
The custodian has authored only its public acceptance interface and has seen no
development prompts/outputs or private evaluation material.

## Frozen parent, runtime and exposure

The official BF16 parent is `Qwen/Qwen3.5-4B`, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Its exact lock-file SHA-256 is
`6058685c1842ee36e24fcb17131ae072332c8cf1e679189e7eaeff4765a2eb8b`.
Every worker authenticates all twelve runtime files, including both original
shards, and binds the file generations actually consumed. Original BF16
parameters and all 48 original FP32 auxiliary parameters retain their original
words and dtypes. All 426 unique language parameters are hashed before and in
`finally` after each forward worker.

The diagnostic profile is greedy, seed 20261108, thinking disabled, one thread,
at most 512 input and 96 new tokens, with the official template and tool schema.
Generation explicitly receives the authenticated model/tokenizer stop union
`[248044,248046]`; the first stop must be terminal. There is no truncation,
sampling, generated-code execution or actual tool execution in these fixtures.
This short CPU diagnostic does not certify native OpenCode/Hermes operation at
262144 context, Q8 K/V cache and 28192 output. Those remain separate gates.

Every official generation uses a fresh model/cache state in a retained,
network-disabled, read-only Docker container, with no GPU and one CPU/10 GiB
limits. Only a new output directory is writable. Shared admission holds an
exclusive lease through confirmed exit; the worker has a 1250-second active
alarm and a 450-second postcheck/exit reserve inside its 1800-second maximum.
Host memory/disk reserves are checked during execution. No other job is stopped
or reconfigured to obtain resources.

The tool fixture and its cues are exposed development data. HE9, the prior
fourteen checks, inspected public benchmarks and historical client fixtures
remain regressions. None is represented as an untouched holdout.

## Original harness failure and one corrected replay

Producer `4c8cd17144b5f02e47a8a4b76dc80565873714a6` ran four generation
invocations. It reported the correct EOS union but omitted it from `generate`.
Both initial outputs crossed their first chat stop at index 25 and ended at
index 27; both continuations crossed index 25 and ended at index 82, fabricating
later chat turns. The gate correctly ended `STOP_harness`; its exact streams,
capsules, source and verdict remain preserved. No prefix was trimmed or reused.

The [F09 review](../reviews/20261108-direct-chat-stop-repair.md) records the
repair and its regressions. The surgeon reviewed, and the owner adopted, one
disclosed post-fault/pre-corrected-score technical replay. All scientific cues,
criteria, strengths and support limits remained unchanged. The revised total
cap is thirteen official discovery generations: four original, one corrected
tranche of four, then the unchanged five semantic controls. All failed and
engineering costs remain charged against 21600 cumulative worker seconds;
the later diagnostic cap remains 22. Durable same-user claims prevent silently
restarting either tranche from another worktree.

Corrected producer `f0b7bd3674e3516611af786aca7ffc27fdbc7f57` completed two
fresh full episodes. Both spontaneous initial calls were valid; both following
continuations repeated exactly the same parser call. All four outputs contained
26 tokens and ended at their first EOS (index 25). Initial/continuation input
lengths were 317/374. Greedy input/output IDs repeated exactly across episodes.
Every protected parameter remained unchanged. The recomputed gate is
`BF16_duplicate_call_reproduced`, admitting only semantic controls, not weights.

## Five semantic controls

Producer `a0cec123359ceff613e6fc1f3a9088634fb73a07` freezes the compiler,
worker, host controller, protocol, qualification ledger and AGENTS safeguards.
The protocol SHA-256 is
`628b5674bbd951e42f4c182c8678b3ea1ccdcb126b73efc2e02598f669301364`;
the prepared conditions SHA-256 is
`324a91d0c81e5ee85b969bb9b95dffe559bcef9df2aa9d7668680ce2038af6a8`.

| Control | Fixed condition | Required behavior | Input tokens | Result |
| --- | --- | --- | ---: | --- |
| B | Exact failed-result replay | Same parser call and exact tokens/vectors | 374 | PASS |
| S | Success result, exit 0, six passes, no failures | Exact JSON exit 0 / failed 0 | 374 | FAIL: repeated call |
| C | Failed result plus fixed completion cue | Exact JSON exit 1 / failed 2 | 390 | FAIL: correct values in a Markdown fence |
| N | Failed result plus fixed neutral cue | Same parser call | 390 | FAIL: same fenced JSON as C |
| P | Two runs requested; one result completed | Legitimate second parser call | 390 | PASS |

B/S/P each generate 26 tokens; C/N each generate 16. All five end at their first
EOS, with no censoring, parser disagreement or parameter change. The primary
and independent bounded parsers agree on every classification. B/P passes are
matches to experimental predictions, not a coding-accuracy score: B explicitly
predicts the base's bad duplicate call.

C copies the correct numeric object `{"exit_code":1,"failed":2}` but surrounds it
with a Markdown JSON fence, violating the frozen bare-JSON rule. N produces the
identical output token sequence. Thus even relaxing C's wrapper after the fact
would not restore the required completion-specific contrast, and S remains a
decisive falsifier. No wrapper stripping, neutral-cue reclassification or changed
criterion was used to salvage the hypothesis.

The official tokenizer found the first lexicographic C/N padding match at two
and one repetitions of `recorded`. Both cues have identical input length and
corresponding observation positions. No model scores entered the bounded
tokenizer-only search. Each worker authenticates the historical four-record
source bank and recompiles all conditions before its one generation.

Read-only hooks capture all 32 MLP down-projection input/output vectors at three
positions: final assistant input token, final tool-result data token, and prior
call's actual `parser` data token. Full rendered text, IDs, BPE offsets and data
spans prove the locations. Fresh B must match Stage 1's exact input/output IDs
and all 64 assistant-position BF16 vectors. No layer geometry or feature
selection is admitted before all five semantic predictions pass.

An ordinary completed semantic failure irreversibly rejects localization but
still leaves all five completed controls in the denominator. Infrastructure,
censoring, source/integrity or budget failure aborts the remainder and records
attempted-unverified controls separately from unstarted ones. Gate, worker
completion, parameter integrity and final quality are separate fields.

## Engineering qualification and budget

The [integration review](../reviews/20261108-direct-tool-specificity-integration.md)
records the three-position worker, historical/current source separation and
three controller fixes made before any semantic model score. The final Linux
qualification ran **440 tests: 421 passed, 19 skipped**, no failures/errors, in
61.376 seconds. The separate admitted inference container passed **12/12**
synthetic Torch/Transformers checks, including all three capture positions,
cached-token callbacks and interrupted-generation all-426 postchecks, in 7.574
seconds. Twelve inference skips were therefore exercised separately; the seven
remaining skips concern preserved legacy training/PEFT/export APIs.

The original failed work cost 1751.884156 seconds; corrected qualification cost
91.829473 seconds; corrected generations cost 1238 seconds. Two semantic
engineering tranches add 206 conservatively charged seconds. The
[frozen ledger](direct-tool-finalization-cycle2-20261108/SPECIFICITY-QUALIFICATION-LEDGER.json)
therefore charges **3287.713629 seconds and eight official generations before
the semantic tranche**. Each tranche uses the larger of whole wall time and
the sum of rounded step times. The host additionally charges model workers
through resource queries, confirmed lease release and receipt writes.

The five semantic workers add 319, 324, 314, 314 and 334 seconds, respectively:
1605 seconds. The terminal discovery ledger is **4892.713629 charged worker
seconds and thirteen official generation invocations**, including all four
original invalid streams. The 21600-second ceiling is respected. No additional
localization/causal diagnostics were run. Archive collection, offline replay and
publication are post-experiment handling, not extra model generations.

## Independent acceptance and current limits

The [Sol max public interface](../plans/direct-cycle2-independent-acceptance-20261108.md)
requires one eligible immutable candidate and frozen runner/checker before any
private cases: 24 cases in eight families, 48 matched base/candidate generations,
at least 12.5 percentage points net executable gain, zero task regressions,
improvement in at least five families and the preregistered family uncertainty
thresholds. The final runner/checker remain unimplemented prerequisites.
Separate-agent custody under one OS user is documented; no adversarial ACL
boundary is claimed. No-op repeats or the development controls are not final
paired quality evidence.

The S failure rejects the hypothesis before localization. It shows that the
duplicate-call phenotype in this exposed fixture is not restricted to a nonzero
result. It does not establish a replacement general-finalization mechanism or
identify defective weights. No geometry ranking, coordinate selection, activation
intervention or weight deletion is admitted from these observations.

No candidate, independent-final generation, quantized conversion, native client
deployment or Hugging Face model promotion has occurred in this cycle. Independent
quality remains `NOT_RUN` because there is no eligible candidate to pair with the
base. Running the base twice or relabeling these controls would supply no repair
comparison. The Sol max custodian is not invoked on private cases. Any next
hypothesis needs a new reviewed cycle and new finite controls; it cannot reuse
this gate as a pass, weaken its cues or introduce a replacement fitting method.

## Preservation and publication

All old and current source trees, failed preflights, intermediate source
preimages, original and corrected reports, capsules, activation archives,
token journals, retained containers, cost ledgers and durable claims remain on
`.54`. No original artifact/cache cleanup, baseline write, service restart or
provider-route change is authorized or performed. The stopped training unit
remains stopped. Spark Fable Runtime, Bonsai, Moss and unrelated evaluations
remain outside this task's process-control scope.

The [results index](direct-tool-finalization-cycle2-20261108/RESULTS-INDEX.md)
links the machine-readable findings, source freeze, terminal gates, inventory,
qualification logs, offline replay and tamper-negative proof. Its 19,292,280-byte
archive contains 509 regular non-executable members: 460 selected filesystem
evidence/staging files, 47 producer/preimage source snapshots stored as text,
the safety record and manifest. It is a curated portable copy, not a backup of
the model checkpoints or every unrelated file. Every original remains on `.54`.
All listed data/source hashes were verified after restore. The archive SHA-256
also binds its manifest:
`98fc1f7c20f28cad1351d87e7c884e19bf7c02f23d82f45434aa52a586771df6`.

The inventory records sizes/allocation, uid/gid, mode, hardlinks, original paths,
dependencies, usefulness and recovery locations. Atomic publication staging
files are retained; the two generated config/ledger staging prefixes are
specifically ignored by Git and remain inventoried. Nothing is classified as
useless or deleted. Source snapshots in the archive are inspected as data, never
imported or executed by the offline checker.

At the final safety capture, the training unit is `MainPID=0`, `Restart=no`,
`ActiveState=failed` (the preserved stopped state). The runtime checkout remains
`026a7da99073cd9c69f73bc299c8667cc0666a3f`; the previous direct-study checkout
remains `f0f340213dd863d6d780bc52e35b20e4e17a2b08`. Original/corrected Cycle 2
checkouts remain `4c8cd17`/`f0b7bd3`. No process-control command targets Spark
Fable Runtime, Bonsai, Moss, another job or the separate evaluation.

The publication target is **`main`**, using a normal fast-forward push from the
verified previous tip `afae4bf`. After completion, the user explicitly requested
public visibility and comparison with related GitHub projects. Publication
includes a reachable-history and artifact check before changing visibility.
Scientific producers retain their measured commit identities independently of
the later documentation/archival commit; the external delivery receipt records
the final remote tip, visibility and GitHub Actions result.

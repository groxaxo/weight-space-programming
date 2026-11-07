# Qwen3.5-4B semantic repair round — 8 November 2026

[Raw paired evidence](semantic-repair-20261108-round1-paired.json) ·
[Machine-readable summary](semantic-repair-20261108-round1.json).
Named companion files below are retained in the full experiment evidence bundle.

**Measured outcome: not_promoted.** The predeclared development/client
replacement gate fails. Model promotion is a
separate publication decision; it is not implied by a training run or this report.
The original released adapter remains identifiable by its immutable Hub revision.

## Paired results

| Test | Base | New semantic repair adapter |
|---|---:|---:|
| Development code | 7/8 | 5/8 |
| Development tools | 5/6 | 5/6 |
| File edit / opencode | PASS (21.471s) | PASS (20.920s) |
| File edit / hermes | PASS (60.656s) | PASS (61.648s) |
| Two-module project / opencode | PASS (281.157s) | FAIL (1000.001s) |
| Two-module project / hermes | FAIL (257.387s) | FAIL (243.935s) |

Development improvements versus this fresh base: `['half_open_overlap']`.
Regressions: `['escape_fields', 'version_tuple', 'bounded_chunks']`. Single timings do not establish
repeatable speed gains. See `paired-comparison.json` and the per-arm raw results
for executable assertions, traces, independent grading and model attribution.

## Why this round was run

The preceding 28,192-token pair passed three of four actual client trials in
both arms. The released MLP scored 11/14 development cases versus base 12/14.
Its `bounded_chunks` used `isinstance(size,int)` and accepted `True`. Both Hermes
projects failed: generator reuse, duplicate/missing validation and incorrect
dependency/order logic survived the model's edits. Both models also failed the
half-open empty-interval and truthful-failure development cases. The earlier
1,024-token omitted-limit bug was already fixed; this round does not attribute
program correctness failures to that resolved budget defect.

## Fixed method and data

The recipe was committed and pushed as `539ee055084853ee9816e20b9a16fb9f65d7a264` before training. CI run
42407757525 passed. The full `.54` regression suite passed **246 tests, zero
skips**, including actual pinned Docker controls, in 62.947 seconds. The new
single-test fixture exposed the offline runner's incorrect plural-only output
check; the regression failed before that fix and passed afterward.

The run starts from the authenticated released MLP with **192 exactly equal FP32
factors**, a fresh optimizer and the unchanged official base revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. NF4 double-quantized training uses
rank 8, alpha 16,
96 MLP modules, dropout 0.05, learning rate 3e-05,
one epoch, seed 20261108, accumulation eight and max length
1,536. The longest backward probe is 1,535 tokens and has finite gradients.
The saved checkpoint is chosen by lowest held-out loss before native testing.

Replay contains 640 licensed public and 288 original authored conversations.
There are 360 additional records across ten separate semantic
families: strict integer arguments, generator materialization, duplicate names,
original-order priority, empty half-open spans and read/edit/test/stop repairs.
The final split is **1156 train /
132 validation**, with shared family IDs across
all variants and interfaces. Maximum new-record length is
1226 tokens; no truncation is used.

The generator never reads benchmark prompts, hidden/reference answers or client
acceptance source files. It does not contain a chunking function or the scheduler
acceptance solution. Requirements are informed by inspected failures. These
compact development and client tasks are therefore tuning-informed, not unseen
evaluation. Family-disjoint splitting is a useful control, not proof of broad
semantic independence or absence of pretraining contamination.

Each broken and fixed authored implementation was run in the pinned offline
Docker image. File/edit tool feedback is simulated; unittest outcomes are actual
offline observations. Those controls are not measured OpenCode/Hermes scores.

Held-out loss: **0.262717 →
0.116451**. Selected checkpoint:
`/home/op/work/weight-space-programming-20261106/runs/semantic-repair-train-20261108-01/training/checkpoints/checkpoint-74`. Training runtime:
1890.226 seconds; peak allocated
11102416384 bytes. All recipe, data and adapter hashes
are retained in the training manifests.

## Matched runtime and evidence

Both arms use the same official Q8_0 base, native binary/shared libraries,
**262,144 context / 28,192 output default and cap / Q8 K and V**, one slot,
non-thinking instruct sampler and seed 20261106. Conservative input reservation
is 233,952. Explicit smaller development limits remain honored. Full allocation
does not establish quality or recall at full-window length; training length is
a separate limit.

All 177 actual client requests are checked against their effective
native generation limit, including auxiliary calls. Native parse errors: 0.
All actual clients attributed: **True**.
File/project deadlines, immutable tests, approval policy, Hermes turn limit,
Docker containment and installed client versions match between arms. The
profile uses fresh isolated client state on the host, not a new virgin VM per
arm. Historical selected HE9 measurements retain their original 8,192-token
identity and are not relabeled as a new test of this adapter.

Candidate PEFT SHA-256: `cb50b579b5cf44755513a3d257a21f65d64636dedaa4536c24ad52c2d2bcb2d5`.
Native F32 adapter SHA-256: `b5cd95209f69cc7c5c1a06548a4bcfcb84293a90f78fcccd880f710c90db076d`.
Base Q8 SHA-256: `2164b7e8728dc954702d86fe423da3a240964877d3d31b8f6530d38702c38547`.

## Predeclared gate

- `strict_development_win`: **False**.
- `no_base_development_regression`: **False**.
- `bounded_chunks_pass`: **False**.
- `hermes_project_pass`: **False**.
- `all_four_clients_pass`: **False**.
- `valid_complete_evidence`: **True**.

The protocol predates model testing. No acceptance assertions, required
attribution, approval controls, client deadlines or turn limits were relaxed.
`failure-diagnostics.json` preserves saved test output for failures. Any further
recipe must be recorded before testing; every negative result remains visible.

## Reproduction and operations

Use the exact source archive, public input pins and new generator with the
existing authenticated `.54` training, FP32 export and native conversion tools.
The `reproduction/` controller records exact arguments and source hashes. Do not
reuse completed output directories. The corpus build, execution controls,
parent equality proof, checkpoint selection, conversion/export manifests and
per-arm ready/completion/restoration records accompany this report.

The controller is bounded by systemd and uses the existing GPU owner lease and
restoration callbacks. Only known Bonsai is paused. Moss, protected runtimes,
canonical memory and the historical VM disk are preserved. Final restoration
and publication are recorded separately in `final-host-state.json` and
`PUBLICATION.json`; those point-in-time records take precedence over a running
phase's provisional state. No Hugging Face token is transferred to `.54`.

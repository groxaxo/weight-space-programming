# Qwen3.5-4B direct-weight editing

The [Cycle 2 tool-action study](experiments/direct-tool-finalization-cycle2-20261108.md)
reproduces a completed-result duplicate call in two fresh original-BF16 episodes.
Its separately frozen success-result control also repeats the call, rejecting
the nonzero-result-specific hypothesis before layer or feature localization.
The new `ops/direct_tool_specificity54.py` route verifies
the historical producer bank, compiles only fixed exposed conditions, admits the
whole tranche and runs one CPU generation per sandbox. All five completed
semantic controls remain in the denominator even after an ordinary failure;
infrastructure/censoring stops retain explicit unstarted rows. This path does
not apply weights or implement the independent final runner.

The active method is forward-only diagnosis and bounded direct BF16 intervention,
following the [pinned exp8 constraints](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md).
There is currently **no demonstrated 4B direct repair**. The 0.8B reconstruction
and historical LoRA results retain their separate identities. No training,
backward/autograd update, optimizer, fitted probe, loss-fitting procedure or
trained-adapter merge belongs to this path.
Upstream publishes the rules and logs, but omits its agent sandbox/forward-server
harness code. This repository's new 4B broker is a bounded implementation of
the declared boundary, not that unpublished harness.

The core `requirements.txt` remains Torch-free for integrity and CPU protocol
tests. Optional `requirements-direct-inference.txt` records the versions used by
the forward study without PEFT, bitsandbytes, datasets or FLA requirements. It is
not a hash-locked standalone distribution. The measured `.54` runs reuse existing
dependencies read-only rather than installing another environment; a fresh
platform/package build still needs its own runtime and paired validation.

## Implemented CPU contract

`scripts/direct_weights.py` uses only the standard library and NumPy. It imports
the existing safetensors-header validator and directory-fsync helper from
`scripts/pipeline.py`; it never invokes that script's reconstruction workflow.
It does not import Torch, load a model, use a GPU, call a provider or write a
checkpoint. The separate `configs/qwen35-4b-direct.lock.json` authenticates the
official `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`:

- Two official safetensors shards, totaling 9,319,828,096 bytes. The editable
  MLP matrices are BF16; original FP32 auxiliary parameters retain their dtype
  and exact bytes during forward inference.
- Ten required license/config/tokenizer/template/preprocessor/index files with
  exact sizes, SHA-256 and immutable upstream object identities.
- All 32 permitted language MLP down projections with exact BF16 dtype, shape
  `[2560, 9216]` and shard mapping, plus the actual hybrid 32-layer architecture.

The lock is derived from the retained baseline audit proof, whose expected hashes
were checked against immutable upstream metadata. That audit inspected metadata
and headers without freshly hashing the multi-GB blobs. Every `verify-baseline`,
`inspect-tensors` and `validate-patch` invocation **fully rehashes** the required
files before claiming integrity. The official README and other extra snapshot
files are preserved but are not runtime inputs in this lock. Reusing a trained
candidate as the parent is not supported. This lock does not alter the original
`reproduction.lock.json` or transfer any 0.8B coordinates.

```bash
python3 scripts/direct_weights.py verify-baseline --base /exact/official/snapshot
python3 scripts/direct_weights.py inspect-tensors --base /exact/official/snapshot
python3 scripts/direct_weights.py validate-patch \
  --base /exact/official/snapshot --patch /exact/candidate.json \
  --report /existing/private/directory/new-validation.json
python3 -m unittest discover -s tests -p test_direct_weights.py -v
```

The optional global `--lock PATH` precedes the subcommand. `inspect-tensors`
returns authenticated names, dtype, shape, shard, byte offset and size; it is not
a cheap metadata-only integrity claim. Full authentication reads about 9.34 GB,
uses bounded read buffers and writes no baseline files. Symlinks in every path
component are resolved and reported, including snapshot-to-model-local-to-global
blob chains. Handles use `rb`; original permissions, hardlinks and targets are
not changed. Paths are operated in an owner-controlled tree. The tool detects
file changes during each hash but is not a defense against an adversary that can
rewrite trusted code, lock or the directory tree concurrently.

## Sparse JSON format

Version 1 accepts **JSON only**, never pickle, object arrays, Python programs,
trained low-rank factors or a writable checkpoint. Each input is at most 32 MiB.
Duplicate keys, unknown fields, non-finite JSON numbers, duplicate tensors,
duplicate/unsorted/out-of-range positions, booleans/floats masquerading as integer
coordinates or words, non-BF16 shapes/dtypes and undeclared tensors are rejected.

The top-level fields are exactly:

| Field | Contract |
| --- | --- |
| `schema_version`, `format` | Integer `1`, string `qwen35-4b-sparse-bf16-v1` |
| `parent` | Exact `model_id`, full `revision`, SHA-256 of the exact lock-file bytes as `lock_sha256` |
| `provenance` | `hypothesis` and the six record identities listed below |
| `limits` | All six numeric limits in the table below |
| `tensors` | Unique permitted tensor records; an empty list is a no-op control |

`provenance` contains `hypothesis`, `failure_record_sha256`,
`practice_manifest_sha256`, `localization_record_sha256`,
`control_protocol_sha256`, `producer_code_sha256` and
`forward_observation_sha256`. Each hash is lowercase 64-character SHA-256.
These record identities are required declarations; the CPU validator does not
open, grade or establish the scientific truth of those records. The custodian
must retain their exact bytes and review their derivation and exposure boundary.

Each tensor record contains exactly `name`, `shape`, `dtype`, `before_sha256`,
`after_sha256`, `indices`, `before_bits`, `after_bits`, `selected_entries` and
`changed_entries`. `indices` are strictly increasing row-major flattened integer
positions. BF16 words are explicit integers in `[0, 65535]`, serialized as raw
little-endian uint16 when hashing/applying. Every selected before word must match
the authenticated parent. There is no FP32-to-BF16 rounding or silent dtype
conversion in application. Any BF16 exponent pattern denoting NaN or infinity
is rejected; the complete affected original/edited tensor is checked for finite
values. `selected_entries` can exceed `changed_entries`: selecting an unchanged
word counts as selected but contributes zero changed positions.

The full raw tensor byte hashes must match both before and after application.
The implementation copies the original tensor and assigns only declared
positions, preserving all other words. It computes original L2 norm, delta L2
norm, relative L2 delta, maximum absolute delta and selected fraction. Bounded
FP64 diagnostic reductions avoid allocating a full FP32/FP64 model tensor.

| Required limit | Maximum accepted value |
| --- | ---: |
| `max_tensors` | 4 |
| `max_selected_entries_total` | 65,536 |
| `max_selected_entries_per_tensor` | 65,536 |
| `max_selected_fraction` | 0.01 of each affected tensor |
| `max_absolute_delta` | 0.125 |
| `max_relative_l2_delta` | 0.05 |

Candidates may set stricter limits. These ceilings are conservative engineering
bounds, not evidence that an intervention of that size is safe or effective.
The sparse support restriction excludes whole-tensor rescaling. Embeddings,
output head, final norm, attention and vision tensors are outside this version's
allowlist. An all-zero parent with nonzero delta fails the relative-norm bound.

Reports bind the exact input patch bytes and parent identity, distinguish
selected/changed totals and say `quality_evaluation: NOT_RUN`. Invalid inputs
fail with explicit exceptions under normal and optimized Python.

## Trusted forward-worker interface

```python
import direct_weights as direct

lock = direct.load_lock(lock_path)
baseline = direct.verify_baseline(lock, model_directory)
patch = direct.read_json(patch_path)
direct.validate_patch_structure(lock, patch)
# After authenticating the private loaded parameter against this exact parent:
private_words, stats = direct.apply_tensor_patch(
    original_words, patch["tensors"][0], lock, patch["limits"])
```

`original_words` must be a contiguous little-endian NumPy uint16 view with exact
shape. It may be read-only. The returned array is a writable private NumPy copy,
never a shared file mapping. `apply_tensor_patch` independently rechecks tensor
support, before/after hashes, selected before bits, counts and limits, but cannot
authenticate other checkpoint tensors. The trusted worker must check the whole
patch, authenticate the whole parent and apply one declared candidate arm at a
time. Restore and reauthenticate exact originals between arms. Neither declared
provenance nor a function call is an isolation boundary for arbitrary editor code.

`validate_patch` holds at most one selected original tensor (47,185,920 bytes)
and its private replacement at a time, plus bounded reductions and JSON. No full
9 GB derivative is created. `publish_json` writes a private sibling staging file,
fsyncs it, then atomically links the complete new JSON report into place using
no-replace hardlink semantics. This hardlink is between newly created report
files, never to model files. Existing files, directories and dangling symlinks
cannot be overwritten. Complete staging files remain preserved after success
or failure; interrupted writes can leave private staging only. There is no
partial report under the final destination before publication. A post-link
directory-fsync failure can leave a complete destination while reporting failure;
inspect it and choose a new output identity for a retry. Do not delete staging
or outputs automatically.

## Research steps that numeric validation does not complete

The separate forward-only CPU worker and host admission wrapper are first-stage
practice diagnostics. They do not establish a causal candidate, an independent
holdout boundary for arbitrary editor code, a complete paired client evaluation,
quantized deployment or a full-context quality result. Before selecting an edit:

1. Reproduce a coding/tool failure on approved practice data under immutable
   runtime, template/thinking, seed, budget and grader identities. Rule out
   harness, parsing, permissions and budget causes.
2. Localize the mechanism from 4B observations. Predeclare no-op,
   norm/support-matched random, component/layer and strength controls. Do not
   fit weights to a loss or transplant published 0.8B positions.
3. Compare original and patched BF16 with shared per-case seeds and unchanged
   graders. Preserve rescues, regressions, invalid outputs, timeouts and actual
   settings. Previously exposed HE9/client cases remain development regressions.
4. Freeze the candidate and paired protocol before an independently held final
   run. Keep prompts, expected answers, graders and intermediate scores outside
   editor access. Preserve a failed final result; further selection needs a new
   untouched confirmation cycle.
5. After BF16 evidence, treat original/edited quantization and native-client
   acceptance as another pinned paired experiment. A short diagnostic context
   does not prove the 262,144-context/Q8-KV/28,192-output client profile.

Resource admission must preserve Spark Fable Runtime, Bonsai, Moss, other jobs
and the separate evaluation. If memory, CPU or storage is unavailable, queue or
report `NOT_RUN`; do not pause services or delete artifacts. No cleanup, trained
candidate reuse, model promotion, external publication or provider-default change
is implied by a valid patch report. See root `AGENTS.md` and
`research-policy.json` for the active method boundary.

## Bounded .54 practice runner

`ops/direct_edit54.py` admits `verify`, `observe`, `probe` and the fixed `causal`
practice experiment against the official snapshot.
It creates a new run directory and retains the request, source snapshot, mount
identities, invocation, console, Docker state, failure and resource records. It
never controls existing services. `observe` requires 10 GiB worker allowance plus
4 GiB available host reserve; `probe` and `causal` use the same allowance.
`verify` uses 1 GiB plus that reserve. All require
2 GiB free disk. During execution the broker stops only its own named container
if available host memory falls below 3 GiB, disk below 1 GiB, or its deadline is
reached. The current worker maximum is 1,800 seconds; probe requests may use at
most 1,700 seconds, leaving shutdown reserve. Earlier recorded protocols retain
their original larger declared limits; their actual runs completed within 1,800
seconds. These checks cannot guarantee zero shared-host performance impact.

The cached image is pinned by digest. The worker has one CPU, no GPU or network,
no writable root filesystem, no capabilities, no host home or Docker socket,
no swap allowance, and a bounded temporary filesystem. Only the frozen code,
existing dependencies, frozen practice request and twelve resolved model files
are read-only mounts; a separate artifact directory is writable. Sources and
requests are copied and hashed before launch. No image download or cache cleanup
is performed. Containers and failed attempts remain retained.

```bash
# Use the already installed CPU environment on .54, from the isolated checkout.
PY=/home/op/work/weight-space-programming-20261106/.venv-cpu/bin/python
"$PY" ops/direct_edit54.py verify \
  --baseline /exact/official/snapshot \
  --deps /exact/existing/python3.12/site-packages \
  --request /exact/frozen-practice-request.json \
  --output reports/new-verification-id
"$PY" ops/direct_edit54.py observe \
  --baseline /exact/official/snapshot \
  --deps /exact/existing/python3.12/site-packages \
  --request /exact/frozen-practice-request.json \
  --output reports/new-observation-id
```

The initial worker deliberately admits **practice only**, never a patch or final
holdout. Its short CPU profiles have at most 256 input tokens and either 192 or
384 new tokens, greedy non-thinking generation, one thread and seed 20261108.
Changing a profile requires a new frozen protocol/request identity. These are
diagnostics, not the native 262,144-context/Q8-KV/28,192-output acceptance profile.
The surgeon's review identified a historical 290-token answer, so a separately
identified 384-token protocol avoids known 192-token budget censoring. A capped
or incomplete response is recorded as censored, not a localized semantic defect.

`scripts/direct_forward.py` uses inference mode with all parameters frozen and
no optimizer. It loads the official language component, validates the loader's
missing/unexpected keys, then authenticates **every loaded parameter** against
the corresponding original safetensors bytes. Explicit `dtype=bfloat16` in a
library loader can round original FP32 auxiliaries such as `A_log`; the worker
reloads their exact original words and dtype instead of casting rounded values
back. All loaded parameters are hashed again after observation. Vision is not
executed in this text-only diagnostic; its checkpoint bytes remain authenticated.
Installed FLA dispatch may select a GPU kernel even on CPU. This worker explicitly
selects the installed Transformers module's own unwrapped PyTorch reference
functions for gated-delta attention and causal convolution. It verifies their
source module, records source hashes and tests chunk/recurrent consistency on
synthetic tensors. It changes no library files, GPU routing or stored weights.
This declared CPU runtime remains distinct from historical native Q8 inference.
Loaded-parent verification, per-token journals and interrupted observations are
retained even if a later forward step fails.

The worker saves the first-prefill last-prompt-token MLP inputs/outputs for all
32 layers as numeric BF16 uint16 arrays. This position predicts the first output
token. It **does not localize a later faulty code decision**. Two identical greedy
replays establish repeatability, not independent quality observations. A causal
research step needs observations at the actual failure decision, semantic
contrasts and controlled activation interventions before any weight selection.
Neither these snapshots nor a valid patch schema supplies a discovery algorithm.

`ops/grade_direct_practice.py` runs completed exposed regression output in a
separate bounded, no-network Python container. It binds the generated file,
frozen request, exact original assertion source and grader code by SHA-256,
rejects changed prompts and token-censored output, requires the protected
completion exit status, and preserves the container, task files, bounded logs
and Docker state. It never executes generated code on the host. The supported
practice assertions are already exposed; this is not an independent final
evaluator or a defense against an adversarial program that knows the grader.

## Decision-position practice probe

`scripts/direct_probe.py`, launched through `ops/direct_edit54.py probe`, binds a
completed uncensored practice generation and its original request by exact file
hashes. It targets the actual executable `if start >= end:` decision in the
reproduced empty-interval failure. It preserves the patient's original prompt
and generated prefix token IDs, verifies the wrong continuation against the
patient, checks candidate decoding and records canonical BPE differences.
Inputs beyond the separately declared 512-token CPU diagnostic limit are refused,
not truncated. This is unrelated to the native client's supported context limit.

The initial frozen protocol has two forward calls: original prompt and the same
prompt with an explicit valid-empty/reversed-interval semantic cue. Each call
batches the two teacher-forced operator continuations. It scores their joint
conditional log probabilities, reports the next-token argmax and saves numeric
BF16 MLP input/output vectors at the decision position for all 32 layers. These
are same-model prompt contrasts, not base-versus-edited quality tests.

The forced prefix already includes an earlier incorrect docstring definition of
reversed intervals. Thus this probe measures the later guard conditional on that
mistake. A null cue effect cannot rule out an earlier semantic mechanism, and a
positive cue effect does not establish defective weights or a repair.

Optional explicit arms implement self/cue activation replacement in four fixed
eight-layer groups at strengths 0, 0.5 or 1, with measured time admission. They
are **partial controls**: matched-random, irrelevant-position and reverse-transfer
controls are not implemented. No group or weight coordinates are automatically
selected, no parameters are changed, and no independent qualification is admitted.
A mechanism and reviewed non-fitting derivation remain necessary before a sparse
BF16 weight candidate can be proposed.

Each probe retains request/tokenization proofs, per-arm observations and scores,
progress/interruption records, and `result.json`. All loaded parameter hashes
must match before and after. The host controller validates complete result
bindings; a bare successful exit cannot substitute for evidence. Incomplete work
has `complete: false`, and all probe results have `quality_evaluation: NOT_RUN`
and `weights_edited: false`. The [conditional localization plan](plans/direct-localization-20261108.md)
records the additional controls and practice-family replication still required.

## Finite activation causality study

Schema 2 of `direct_probe.py` targets the earliest incorrect docstring assertion,
before the executable guard. It preserves the original patient's actual prefix
and compares baseline, semantic cue and a frozen length/format control. The cue
and control must have identical token lengths and decision positions. Replayed
baseline and diagnostic thresholds are mandatory; an explicit prompt cue fixing
an answer still does not show a weight defect.

`scripts/direct_causal.py`, admitted as `ops/direct_edit54.py causal`, implements
a separately frozen, activation-only experiment. The six donor files are
`request-proof.json`, `result.json`, `tokenization-proof.json`,
`cue-decision.npz`, `length_control-decision.npz` and `runtime-identity.json`.
Stage 2 additionally binds `stage1-result.json`. They are exclusively copied,
hashed, bounded and mounted read-only; numeric NPZ members, dtypes, dimensions,
finite values and ancestry are validated without pickle. The historical runtime
record distinguishes a separately observed Torch version and controller pins
from a per-process binary closure. Current package/kernel/source identities and
all 426 parameter hashes must match. A fresh neutral replay must match raw
branch log probabilities within 1e-6 and all decision vectors bit for bit.

At each of 32 MLP down projections, the frozen direction is the semantic-cue
output minus the equal-length control output at the decision token. Additions
use FP32 arithmetic and round back to BF16; parameters never change. Stage 1 has
exactly six forwards: neutral replay, a zero-strength hook and four consecutive
eight-layer groups. A group qualifies only with greedy `>`, positive correct
minus incorrect joint margin and at least a two-nat increase. The greatest
increase wins; ties within 1e-6 select the smaller group index. No qualifier ends
the hypothesis.

Stage 2 has exactly eight forwards for that winner: neutral replay, full and
half direction, two sign-randomized directions with fixed seeds 20261108 and
20261109, full direction at the assistant code-fence `python` token, semantic
replay and reverse transfer. Random, position and reverse controls must match
the actual post-BF16 per-layer norm and changed-coordinate count within 5%.
Full effect must reproduce, half effect must lie strictly between zero and full,
random/off-target effects must retain `>=` with less than 25% of the full margin
change, and reverse transfer must turn the semantic answer back to `>=` with
negative margin. No alternative seeds or positions are searched after results.

Both stages have at most 1,700 seconds including postchecks inside the 1,800-second
host deadline. Incomplete arms or failed replay/control checks stop weight work.
Even passing both stages establishes only a candidate information carrier;
separate read/write geometry, a non-fitting sparse derivation, practice-family
replication and independent paired model evaluation are still required.
The [paired acceptance boundary](direct-paired-acceptance.md) distinguishes
exposed regression, untouched custody, BF16 evidence and native quantized-client
acceptance. Its unimplemented or unrun stages must remain explicit.


## Audited execution integrity

The [eight-discrepancy repair report](reviews/20261108-direct-harness-fixes.md)
defines the current worker contract. Full SHA-256 authentication is bound to
file generations at consumption and loading; grading and worker requests bind
captured input bytes before execution. All direct brokers share a persistent
host/user admission lease held through confirmed exit. Unknown orphan state
fails closed; do not unlink the lease or interrupt another job.

New results use `evidence_schema_version: 2`, separate from request schemas.
Host acceptance requires the frozen producer, full baseline/loaded-parent
proofs, the exact 426 protected parameter names and matching before/after
hashes, tokenization, ordered requested arms, raw score files and safe numeric
observations. It recomputes diagnostic gates. Interrupted observations retain
all-parameter postchecks and incomplete evidence. Contrast completion alone
cannot substitute for requested controls; EOS at the cap is distinguished from
censoring without EOS. Historical raw records retain their original schema and
producer identity. No artifact is rewritten to pass the newer verifier.

This is an execution-integrity contract. The closed 8 November hypothesis is
still `STOP_controls_failed`; new paired model/client quality remains `NOT_RUN`.


## New fixed tool-episode reproduction

The [Cycle 2 plan](plans/direct-tool-finalization-cycle2-20261108.md) is a new
completed-action hypothesis, not a continuation of the closed interval study.
`ops/direct_tool_preflight54.py` first authenticates the parent/tokenizer and
checks the exact historical system/tool schema, template, argument normalization,
actual token lengths, parser agreement and call-ID invariance without loading
the language model. The first failed preflight and its source remain preserved;
it exposed a fast-tokenizer mapping/list comparison bug before any BF16 score.

`ops/direct_tool_cycle54.py` then submits the frozen `tool-observe` requests to
`ops/direct_edit54.py`. One worker performs one generation, with fresh model/cache
state, 512 input/96 output limits, greedy decoding, thinking false, one CPU,
10 GiB and no GPU. These are short diagnostic budgets. The actual initial call
must be valid before its hash-bound source/request/result/proof/observation
capsule can enter a continuation. No tool action generated by the model is
executed on the host; the exposed fixture returns fixed bytes. Exact raw decoded
special tokens are retained, all 426 parameters are checked in `finally`, and
host verification recomputes the two parser decisions. A tokenizer wrapper is
unwrapped before comparing actual IDs; a wrapper mismatch cannot be relabelled
a neural failure.

The exact frozen protocol is
`configs/direct-tool-finalization-cycle2.protocol.json`. Two valid episodes finish
even when the first continuation is correct; a correct continuation rejects this
specific BF16 failure hypothesis. No candidate exists unless the complete later
semantic, activation, sparse-mechanism and direct-weight controls also pass.
Independent final acceptance belongs to the separate
[Sol max custodian](plans/direct-cycle2-independent-acceptance-20261108.md), with
no private prompts or answers exposed to selection. That interface is a later
prerequisite, not an implemented or completed final run.


## First-stop correction and preserved failed tranche

The first tool-episode tranche is `STOP_harness`: it advertised the correct
model/tokenizer EOS union but passed neither explicit stop list to generation.
The native route already used both `<|endoftext|>` and `<|im_end|>`. The raw
continuation crossed its first assistant end, generated a simulated user/tool
result and a later assistant answer. Those bytes and their original verifier
remain preserved. Neither the prefix nor the suffix is a valid completed trial.

The [separate review](plans/direct-tool-finalization-cycle2-stop-repair-review-20261108.md)
and corrected protocol freeze one technical replay with fresh capsules. The
same authenticated EOS union now controls generation and completion metadata;
the host independently rejects any tokens past the first stop. Internal chat
control markers also stop tool classification before parsing. No output is
trimmed to manufacture a pass. New regressions exercise requested versus
reported EOS, ignored first stops, host verification and fabricated chat turns.

The amended discovery cap is thirteen generation invocations (original four,
corrected four, unchanged five semantic controls), while the 21,600 charged
worker-second and 22-diagnostic ceilings remain. Original costs and corrected
qualification costs are captured before model execution. One durable replay
claim is retained; another corrected harness fault ends this cycle. All earlier
scientific controls and the prior interval-study failure remain unchanged.

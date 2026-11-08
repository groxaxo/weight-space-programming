# Qwen3.5-4B direct-weight study — 8 November 2026

This study restores the original no-training research boundary. It uses the
official unchanged Qwen3.5-4B checkpoint, GPT-6.1 Sol with `xhigh` reasoning as
the surgeon, and bounded CPU-only execution on `.54`. No LoRA, training,
backward pass, optimizer, fitted probe or model-weight update was performed.
**No independently qualified 4B direct repair exists yet.**

## Objective and original divergence

The [pinned exp8 method](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md)
excludes training, gradients and optimizers. The original lab commit `0be1a86`
reconstructs an already published 0.8B patch; it does not discover new patches.
Its indices cannot be transferred to the 4B architecture.

The upstream release also explicitly omits the original agent sandbox and
forward-server harness implementation. Its prompts and rules are public. The
new bounded broker implements a reviewed 4B research boundary from those rules;
it is not the original private harness or a reproduced general discovery system.

The recovered project record first announces training targets on 6 November at
22:19 NZDT, then QLoRA at 22:59. Commit
[`138ca32`](https://github.com/groxaxo/weight-space-programming/commit/138ca32c115fbc8ff7b17ee290230af3beb693de)
formalized that substitution with PEFT, backward/optimizer operations and Trainer.
Later user instructions did authorize adapter work, but the method remained
different from upstream weight-space programming. Those results and artifacts
are preserved as a historical track, not relabelled as direct editing.

The latest completed legacy continuation regressed from the paired base's 12/14
development and 3/4 client passes to 10/14 and 2/4. The subsequent training run
was stopped after logged step 22/172 without a new checkpoint. It remains stopped.
The prior supervised corpora came from OpenCodeInstruct, Hermes function-calling
data and authored repairs, not exp8 practice data. The audit found 128 stale raw
prompt hashes; all 1,528 prepared hashes were correct. Raw historical material was
not silently rewritten, deleted or reused as a new training corpus in this study.

## Implemented controls

- Root `AGENTS.md` and `research-policy.json` define the active no-training
  objective, user-selected surgeon, data custody, no 0.8B index transfer,
  preservation, worker limits and independent paired qualification.
- Legacy supervised-data preparation, training, adapter selection/export,
  GPU/service supervisors, the old pipeline smoke trainer and VM replay routes
  refuse active execution. The old service launcher's start/stop actions are
  blocked; read-only status remains available. No CLI or environment override
  resumes training. Preserved historical functions remain inspectable and have
  synthetic regression coverage under explicit mocks.
- `scripts/direct_weights.py` adds a separate 4B lock and strict sparse BF16
  contract. It authenticates all required files, resolves shared blob symlinks,
  validates exact parent/shape/dtype/words/hashes/counts/bounds, and applies a
  tensor only to a private memory copy. Numeric validity is not quality evidence.
- `ops/direct_edit54.py` freezes sources and requests, binds exact model files
  read-only, admits resources and validates worker evidence. Workers have one
  CPU, no GPU/network/home/socket access, a read-only root, bounded scratch,
  memory/swap/PID limits and retained artifacts. The current maximum is 1,800
  seconds; a probe leaves postcheck/shutdown reserve inside that maximum.
- Generated code is graded only in a separate capped Python container. The
  practice grader retains source, original assertions, hashes, logs and state.
  Container startup errors cannot be mistaken for model outcomes, and an early
  successful exit cannot satisfy the test-completion sentinel.

The [method documentation](../direct-weight-editing.md) describes the APIs and
remaining research stages. Workflow guards are not a sandbox against arbitrary
host Python or a proof that arbitrary numerical programs avoided training.

## Baseline fidelity

Official parent: `Qwen/Qwen3.5-4B`, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
The lock SHA-256 is
`6058685c1842ee36e24fcb17131ae072332c8cf1e679189e7eaeff4765a2eb8b`.
Both shards, totaling 9,319,828,096 bytes, and ten required sidecars were freshly
authenticated against their locked content hashes. The architecture has 32
language layers and BF16 `[2560, 9216]` MLP down projections.

The official checkpoint also contains FP32 auxiliaries. The generic BF16 loader
rounded 48 such language parameters: `A_log` and the linear-attention norm weights
in 24 layers. The new
worker reloads their exact original words and dtype instead of casting rounded
values back. Every one of the 426 loaded language parameters is authenticated
against original safetensors bytes and hashed again after inference. Protected
embeddings/head/norm parameters are included. Vision checkpoint bytes are
authenticated; vision inference is not tested in this text-only study.

Installed FLA dispatch selected a GPU-only kernel on CPU. The isolated worker now
explicitly uses the installed Transformers pure-PyTorch reference functions for
gated-delta attention and causal convolution, recording their source hashes.
Synthetic chunk/recurrent consistency tests passed. No dependency files, GPU
routing or existing inference services were changed.

## Reproduced base-model failure

The first practice target is the already exposed `half_open_overlap` development
case. It is not a new HumanEval run, a replacement for selected HE9, or an
untouched holdout. Two identical greedy, non-thinking CPU replays used 256 input
and 384 new-token limits, a supported full model configuration and frozen source.
Both completed at EOS after **294 tokens**, in 564.55 and 563.58 seconds.

Both returned the same executable code, with solution SHA-256
`8258227f21d4fe6cdd73b6ad72e6f2f18dad3c24523b5c5f9f58c0a6b403573e`.
Its `validate_interval` uses `start >= end`, rejecting the valid empty interval
`(2, 2)`. Separate execution of the exact exposed assertions failed in both
replays, without timeout or token censoring. Original assertion/checker source
and the generated answer were preserved; no post-generation code repair was used.

The replays have identical output token IDs and identical finite MLP prefill
vectors. All 426 parameter hashes match before and after. These two identical
greedy trials establish repeatability, not two independent accuracy observations.
The initial last-prompt-token vectors cannot locate a later code decision.

## Conditional decision probe

Sol implemented a practice-only decision probe at the actual executable guard.
It reuses the patient's 152 generated prefix tokens, preserves original prompt
IDs and verifies candidate tokenization against the actual answer. Baseline and
semantic-cue contexts require 207 and 233 forward input tokens respectively,
with no truncation. Candidate sequences are `[835, 809, 25]` for `> end:` and
`[2518, 809, 25]` for `>= end:`.

| Context | Correct minus incorrect joint log probability | Next greedy operator | Forward seconds |
| --- | ---: | --- | ---: |
| Original prompt | -5.5822637392 | `>=` | 203.79 |
| Explicit valid-empty/reversed semantic cue | +0.3747124368 | `>` | 232.79 |

Both calls completed, and all loaded parameter bytes remained unchanged. This
is a conditional prompt effect. No weights were edited, no unconditional coding
gain was measured, and no independent paired quality test was run.

The forced prefix already contains an incorrect docstring definition of reversed
intervals before the executable guard. The probe therefore does not identify the
first causal error. The cue also adds 26 tokens, changing positions and generic
prompt context. Activation donors cannot be treated as a localized semantic
repair until those confounds and causal controls are addressed.

## Preregistered length/format control

The neutral text was selected using tokenizer counts only, before its model
score. It contains no interval/comparison/equality/validation/error/correctness
advice. The first finite text list had no exact length match and was retained;
the first text in the second predefined list matched 233 input tokens and
decision position 230:

> For presentation, use a plain style. Prefer descriptive names, consistent indentation, and ordinary spacing throughout the body of the function.

This is a length/format control, not provably semantically inert text. The frozen
diagnostic gate requires the repeated baseline's branch token log probabilities
to agree within 1e-6, the control to retain greedy `>=` with its margin within
1 nat of baseline, and the semantic cue to retain greedy `>` with at least a
2-nat margin advantage over the control. These are practical research thresholds,
not statistical significance or independent model qualification.

The control gate **passed**. The repeated baseline's maximum per-token log
probability difference was exactly 0. The control kept `>=` with margin
-5.5641705991, only 0.0180931400 nats from baseline. The semantic cue's advantage
over it was 5.9388830359 nats. No replacement neutral wording was tried after
seeing this model result. This supports a semantic-specific conditional effect
relative to this length/format control, not a weight defect.

## First incorrect semantic assertion

A separately reviewed schema2 protocol moves the target to the first incorrect
`>=` in the docstring, before the executable guard. It preserves 97 actual
patient prefix tokens and scores `> end).` versus `>= end).`. Input lengths are
152 for baseline and 178 for both semantic cue and fixed length control. All
canonical/replayed token sequences match. A baseline replay mismatch would stop
the remaining contexts before localization.

| Context at first wrong definition | Correct minus incorrect joint log probability | Next greedy operator | Forward seconds |
| --- | ---: | --- | ---: |
| Original prompt | -4.1846123856 | `>=` | 145.59 |
| Semantic cue | +6.6345427740 | `>` | 175.06 |
| Fixed length/format control | -3.9133149369 | `>=` | 174.65 |

All three calls completed with all 426 parameter tensors unchanged. The length
control differs from baseline by 0.2712974486 nats; semantic advantage over it is
10.5478577109 nats. All six preregistered diagnostic checks pass. The result still
reports `activation_screening_admitted: false`, `weight_candidate_eligible: false`
and `quality_evaluation: NOT_RUN`: causal intervention, practice-family controls
and weight geometry remain separate research steps. No coordinate selection or
permanent parameter change follows automatically from a conditional prompt effect.

## Fixed causal intervention and controls

The Sol `xhigh` surgeon implemented a separate activation-only worker. The
[frozen protocol](direct-weight-20261108/CAUSAL-PROTOCOL.json) hashes to
`3fbedac088e9e468c63430d6a318da9c85793268171f159b6c0e4229128878e4`.
The worker source hashes to
`c55175a3c1a26c8c03221df28bbd2be683c05c0519e0ceb098eb4e59b49782cf`.
These identities precede the screen's model scores.

Each layer's frozen direction is semantic-cue MLP output minus equal-length
control output at the earliest wrong decision. Interventions add that FP32
direction to the current activation and round to BF16; they do not change
parameters. Stage 1 has exactly six calls: a fresh neutral replay, a zero-strength
hook and four fixed consecutive eight-layer groups. A group requires greedy
`>`, positive joint margin and at least two nats of improvement. Selection is
the greatest improvement, breaking ties within 1e-6 by smaller group index.
No qualifying group stops the hypothesis.

The screen completed all six calls in **1,138.768 seconds**. The fresh neutral
and zero-strength hook both reproduced margin -3.9133149369, with zero raw
log-probability difference and all 64 input/output vectors bit-identical.

| Fixed layer group | Joint margin | Increase over neutral | Greedy operator | Qualifies |
| --- | ---: | ---: | --- | --- |
| 0: layers 0–7 | -3.6591939296 | +0.2541210074 | `>=` | No |
| 1: layers 8–15 | -3.6588315105 | +0.2544834265 | `>=` | No |
| 2: layers 16–23 | +1.2290038019 | +5.1423187388 | `>` | Yes |
| 3: layers 24–31 | -3.1422004812 | +0.7711144557 | `>=` | No |

All 426 parameter hashes matched afterward. The fixed rule selected group 2;
no adaptive subgroup or coordinate was selected. The Stage 1 result SHA-256 is
`d76e7cc0309d01ebe348f6f1e78836718b6aea45558525f54b275db988cf31c8`.
This establishes a repeatable conditional activation effect only.

Only a qualifying group admits the fixed eight-call second stage: fresh neutral,
full and half direction, two fixed sign-random controls, full direction at the
verified assistant code-fence `python` token, semantic replay and reverse
transfer. Actual post-BF16 per-layer norm and changed-coordinate count must
match for random, off-target and reverse controls within 5%. Full transfer must
reproduce; half must have a strictly intermediate effect; random/off-target
controls must retain the wrong operator with less than 25% of the full margin
effect; reverse transfer must restore the wrong operator with negative margin.
No seed, group, position, dose or threshold is changed after seeing results.

Before intervention, all six donor files are bound by exact hashes and numeric
structure. The runtime record explicitly distinguishes historical observations
from a complete binary attestation: Torch was observed in retained practice
output, while Transformers/NumPy were controller pins. Current package versions,
kernel/source identities and all 426 parameters must match, followed by a fresh
replay matching raw token log probabilities within 1e-6 and all 64 input/output
decision vectors bit for bit. Stage 2 additionally binds and revalidates the
Stage 1 result and selection rule. Failed or incomplete evidence stops weight
work. Even passing both stages identifies a candidate information carrier;
it does not establish defective weights or eligible sparse coordinates.

Stage 2 completed all eight calls in **1,485.254 seconds**, with exact neutral
and semantic replay, exact full-transfer reproduction and all 426 parameters
unchanged. The frozen gate **failed: 9 of 10 checks passed**.

| Stage 2 arm | Joint margin | Next greedy operator |
| --- | ---: | --- |
| Neutral replay | -3.9133149369 | `>=` |
| Full semantic direction | +1.2290038019 | `>` |
| Half direction | -1.5280531909 | `>=` |
| Random signs, seed 20261108 | -4.1600832585 | `>=` |
| Random signs, seed 20261109 | -3.0438014725 | `>=` |
| Off-target code-fence position | -3.7868046199 | `>=` |
| Semantic replay | +6.6345427740 | `>` |
| Reverse transfer into semantic context | +0.8175669323 | `>=` |

The failed check is `reverse_incorrect`. Reverse transfer **did** change the
immediate operator to `>=`, but the complete scored `> end).` continuation still
had a positive correct-minus-incorrect margin. Joint scoring also includes the
conditional probabilities of ` end).`; it need not agree with the first-token
argmax. The preregistered reverse rule required both the wrong greedy operator
and a negative joint margin. Its next-token result alone cannot satisfy that
rule. All random, position and reverse magnitude/support checks passed, as did
full reproduction, dose order and random/position specificity.

The result is `STOP_controls_failed`, not a 90% qualification or permission to
tune a stronger reverse dose. It supports conditional influence of the group;
it neither proves absence of causality nor identifies defective weights. The
Stage 2 result SHA-256 is
`3b4a8c7046d906a9cbe75bc57aa67cdaef51ac238996bffd0132333779fa4804`.
A separate arithmetic audit recomputed every recorded joint score, fixed group
selection, ancestry, parameter identity and control verdict without importing
the producer's gate. That audit is integrity verification, not an independent
model-quality evaluation.

The selected [surgeon's final interpretation](direct-weight-20261108/SURGEON-CONTROL-REVIEW.md)
confirms the stop. The [proposed mechanism gate](../plans/direct-mechanistic-gate-20261108.md)
remains **NOT_RUN** because its prerequisites failed. No layer subdivision,
feature selection, geometry-based candidate selection, strength tuning or
permanent weight edit followed this result.

## Validation and retained failures

On Linux `.54`, the full CPU and pinned-Docker source suite ran **360 tests in
62.958 seconds: OK, 14 skipped**. Seven skipped checks require the historical
training stack and remain intentionally unrun. The other seven forward-runtime
checks passed separately in bounded CPU inference containers: five original
dtype/loader/reference-kernel tests, one tiny probe test and one tiny additive
causal test. The causal actual-tokenizer/runtime preflight and all 21 causal
contract/host/tiny tests passed with zero skips. These are source/runtime
controls and synthetic execution checks, not actual agent-quality trials.

The [Linux transcript](direct-weight-20261108/linux-cpu-regression-transcript.txt)
and [causal preflight proof](direct-weight-20261108/causal-preflight-proof.json)
are retained with the [practice records](direct-weight-20261108/README.md).
A restricted Mac full-suite attempt failed at Linux-only `renameat2` and blocked
loopback socket operations; it is retained and is not reported as a full pass.
The supported Linux verification above resolves the target-platform check.

Earlier failed attempts are preserved: worker UID lookup, original FP32 dtype
coercion, GPU-only FLA CPU dispatch and loader metadata sets. The first completed
practice worker then succeeded but its controller failed while serializing
Docker's top-level array. Its original failure was preserved; a new recovery
record authenticated the completed worker and a separate full baseline verify
confirmed the controller repair. The first schema2 request was refused by an
old host schema1 check before any container/model started; a new typed schema
regression and new request identity fixed admission. No failed record was
rewritten into a pass.

## Preservation, limits and release status

All original model files, retained historical artifacts, study failures, source
snapshots, observations and reports remain preserved. No cleanup was performed.
The preceding inventory has 206 preservation rows; filesystem ownership and
incomplete process references do not prove exclusive ownership or disposability.
Shared Hugging Face blob targets must not be cleared. Available disk did not
support another full BF16 checkpoint copy, and the GPU remained occupied.

No Spark Fable Runtime, Bonsai, Moss or separate evaluation job was stopped,
restarted, reconfigured or rerouted. The existing master checkout was left intact;
new work uses `/home/op/work/weight-space-programming-direct-20261108`. CPU work
uses shared resources under explicit limits; zero performance impact is not a
claim. Old LoRA runs and releases keep their historical identities.

The native acceptance target remains 262,144 context, Q8 K/V and 28,192 output.
Short CPU diagnostics do not establish that profile's quality. Full BF16
base-versus-edit development controls, an untouched independently held paired
evaluation, selected HE9 regressions and actual OpenCode/Hermes qualification
remain required after a justified candidate exists. Optional group activation
replacement remains a historical partial interface; the new fixed additive
control study has its own protocol and result. Practice-family replication,
specific read/write geometry and a reviewed non-fitting sparse derivation remain
necessary before a weight candidate, regardless of any activation result.

No justified sparse weight candidate, independent win, model promotion or new
Hugging Face model release is established by this study. Training remains stopped.

| Acceptance stage | Outcome in this study |
| --- | --- |
| Official parent integrity and reproducible exposed failure | Passed |
| Matched semantic diagnostic and group sufficiency | Passed on this practice case |
| Complete fixed causal control gate | Failed reverse joint-margin requirement |
| Read/write mechanism and sparse BF16 candidate | NOT_RUN; no eligible candidate |
| Base versus edited BF16 development pair | NOT_RUN; no candidate |
| Independent untouched paired final | NOT_RUN; no candidate |
| New candidate selected HE9 and OpenCode/Hermes pair | NOT_RUN; no candidate |
| New direct model/Hugging Face publication or promotion | Not performed |

The [machine-readable findings](direct-weight-20261108.json) and
[frozen records](direct-weight-20261108/README.md) preserve the positive evidence
and failed gate together. The engineering boundary is implemented and tested;
the model-repair objective remains unachieved.


## Engineering follow-up: eight harness discrepancies

The [subsequent repair report](../reviews/20261108-direct-harness-fixes.md)
records fixes and regressions for authentication at consumption, frozen grader
inputs, serialized host admission, interrupted-observer postchecks, all-arm
completion, donor-gate recomputation, complete host evidence and EOS at the
output cap. New host evidence has a separate version 2. Original protocols,
raw results, source hashes and negative outcomes are preserved under their
original identities; they are not rewritten to satisfy the newer verifier.
The repairs required no new Qwen inference and do not alter this study's
`STOP_controls_failed` result. Independent paired quality and a justified weight
candidate remain `NOT_RUN`.

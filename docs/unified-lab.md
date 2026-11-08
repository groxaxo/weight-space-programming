# Unified forward-only weight-space lab

`python scripts/wsp_lab.py` is the common entry point for the compatible logic
from the [six-repository source review](reviews/20261108-related-direct-editing-repositories.md).
It combines fixed activation interventions, retained evidence, bounded execution,
frozen diagnostic grading, paired development metrics, numeric patch inspection
and historical replay. No external framework is imported or vendored, and no new
runtime dependency is added. The source catalog pins the inspected upstream files.

This is an engineering integration, not a new successful Qwen repair or an
automatic weight-discovery algorithm. The original BF16 model remains unchanged.
The previous interval and tool-finalization hypotheses stay closed. No official
4B generation, private evaluation or training is launched by integrating the code.

## What is combined

| Source | Compatible logic implemented here | Boundary retained |
| --- | --- | --- |
| exp8 logs | Original-parent identity, separated editor/forward worker, fixed budgets and retained records | No 0.8B coordinate transfer or claim of reproducing its unpublished harness |
| weight-patching | Baseline, no-op, strength, random/location controls; rescues/regressions; exact-output locality and optional KL | No fitted gate/probe, LoRA baseline, parameter search or `all` training workflow |
| nnsight | Explicit module/position interventions and prefill/cache separation | Activation effects alone do not identify defective weights |
| pyvene | Serialized fixed zero/replace/add operations and removable hooks | No trainable intervention or learned rotation |
| ROME | Separate MLP input/read and output/write observations; retained original parameters | No optimized target vector or backward/Adam path |
| MEMIT | Bound parent/update identities and matched comparison accounting | No optimized activation target or inferred multi-layer weight update |

The [catalog](../configs/wsp-lab-sources.json) and review retain exact source URLs,
revisions and blob identities. These are scoped reimplementations of compatible
principles, not a merge of the six repositories' full algorithms or dependencies.

```mermaid
flowchart LR
  A[Reviewed design and exposed case] --> B[Freeze protocol and source identities]
  B --> C[Plan eight fixed controls]
  C --> D[Shared .54 admission and read-only sandbox]
  D --> E[One fresh forward worker per arm]
  E --> F[Raw tokens, BF16 vectors and parameter postchecks]
  F --> G[Frozen diagnostic grader and paired metrics]
  G --> H[Engineering report]
  H --> I[Separate causal mechanism and candidate review]
  I --> J[Independent final custody and native acceptance]
```

The last two stages are separate requirements. This integration never marks them
passed, admits a weight candidate or promotes a model.

## One CLI

```bash
python scripts/wsp_lab.py catalog
python scripts/wsp_lab.py prepare \
  --id NEW_REVIEWED_STUDY \
  --case /absolute/path/case.json \
  --intervention /absolute/path/intervention.json \
  --design /absolute/path/reviewed-design.md \
  --prior-worker-seconds PREVIOUS_CHARGED_SECONDS \
  --output /existing/private/directory/new-protocol.json
python scripts/wsp_lab.py plan --protocol /absolute/path/new-protocol.json
```

`prepare` serializes a diagnostic protocol; it does not build a supervised corpus,
fit a probe, load weights or start training. `plan` launches no process. Source
changes after preparation invalidate the protocol and require a new review; do
not rewrite a completed experiment's source identity or restart its study claim.

Only an authorized new study with a scientifically justified design should use
the heavy route, on `.54`:

```bash
python scripts/wsp_lab.py run \
  --protocol /absolute/path/new-protocol.json \
  --design /absolute/path/reviewed-design.md \
  --baseline /absolute/path/official-snapshot \
  --deps /absolute/path/existing-python-site-packages \
  --output /absolute/path/new-run-directory
```

Other integrated tools:

```bash
python scripts/wsp_lab.py compare --manifest /path/frozen-manifest.json --trials /path/attributed-trials.json
python scripts/wsp_lab.py patch-structure --patch /path/candidate.json
python scripts/wsp_lab.py replay-cycle2 --evidence /path/safely-extracted-cycle2-evidence
```

`patch-structure` checks the existing sparse JSON format only. It does not freshly
authenticate the model, validate causal provenance, apply a model edit or create
a candidate. Full numeric validation remains in `direct_weights.py`; scientific
candidate admission remains separate. `replay-cycle2` recomputes the preserved
negative result using the unchanged historical thirteen-source bank.

## Frozen inputs and controls

A case has `id`, `messages`, `tools`, and an optional `assessment`. The assessment
has `kind` (`exact_text` or `json_exact`), `expected`, `family` and `locality`.
Expected answers are retained as exposed grading data and are never passed to
the chat template. With no frozen assessment, diagnostic grading is `NOT_RUN`.
Strict JSON rejects fences, duplicate/extra keys, nonfinite values and mismatched
numeric/boolean types. Executable coding grading is not implemented by these
literal graders.

Supplied tool calls/results are frozen counterfactual input data. IDs and argument
objects are checked and associated, schemas cannot fetch external references,
and no emitted tool is executed. Supplied history is not proof that the model
spontaneously performed an initial tool action. Use the separately authenticated
episode workflow when that stronger evidence is required.

The intervention names one language `down_proj` layer, its `input` or `output`,
a declared prompt token position (`-1` means the last), a distinct off-target
position, sorted unique activation indices, and `zero`, `replace` or `add` with
explicit finite BF16 words. No feature/layer is selected automatically. Token
indices are bound to the actual full input IDs; a design needing semantic-span
claims must separately justify that those indices refer to the intended data.

Eight arms are fixed before scores:

1. Original baseline.
2. Numeric no-op at the selected site.
3. Full declared activation treatment.
4. Half of the actual rounded treatment delta.
5. First fixed hash-seeded sign control.
6. Second fixed hash-seeded sign control.
7. Same declared operation at `(layer + 1) mod 32`.
8. Same declared operation at the frozen off-target prompt position.

Random controls derive from the actual BF16 treatment delta, with no fitting,
resampling or score search. Applied support and L2 must match within 5%; a
degenerate control equal to the treatment/no-op is rejected. This check is not
a universal causal sufficiency test. Reverse transfer, semantic specificity and
a justified read/write mechanism are still required before any weight derivation.

Hooks operate once during prefill and leave cached decoding calls alone. They
copy activation tensors and never write a parameter. Before/after words and
actual norms/counts are saved as bounded, non-object NPZ data and independently
recomputed by the host verifier. No clipping or overflow fallback is performed.

## Execution and evidence

`ops/wsp_lab54.py` reuses `ops/direct_edit54.py`'s existing shared lease, resource
checks, immutable image and hardened container profile. It retains each worker
and holds admission through confirmed exit. All code, dependencies, twelve model
inputs, protocol and design are read-only; only the fresh output directory is
writable. There is no network or GPU access. No unrelated job/service is controlled.

One worker performs one fresh greedy generation, at most 512 input and 96 output
tokens, thinking disabled, seed 20261109 and one CPU/10 GiB. It passes the exact
official model/tokenizer EOS union and rejects output past the first stop. Inputs
are not silently truncated. The active alarm is 1250 seconds, followed by a
450-second postcheck/exit reserve within the 1800-second worker maximum.

Before starting, all eight worst-case workers plus declared previous charges
must fit the cumulative budget (at most 21600 seconds). Each actual attempt is
charged through resource checks, lease release and receipt writes. A durable
same-user study-ID claim prevents accidental re-execution. It must never be
deleted to obtain a retry. Missing resources, censoring, changed source, failed
evidence or a bad no-op stops the remainder, retaining explicit unstarted and
attempted-unverified arms.

The shared loader authenticates the official BF16 parent and restores all original
FP32 auxiliaries. All 426 unique language parameters are checked before/after;
postchecks still run after handled preparation/generation, hook-cleanup or
partial-save errors. Forced process termination can leave incomplete evidence,
which is rejected. The first no-op must replay both tokens and all 64 saved MLP
vectors before treatment is launched.

The new twenty-one-file source manifest extends the evidence boundary without
altering any of the thirteen historical producer files. It includes policy,
catalog and new engine/runner/metric sources. CLI guards and source hashes are
workflow evidence, not an adversarial sandbox or proof that arbitrary upstream
analysis never fitted a tensor. The design and provenance still require review.

## Metrics and honest outcomes

For a verified run with a frozen literal assessment, the host derives grades from
raw baseline/treatment outputs. `development_metrics.candidate_kind` explicitly
says `activation_treatment_only`; no weight candidate exists. It reports paired
rescues/regressions, net change, families and exact-output locality using the
fixed denominator. A missing/censored trial cannot become an observed rescue.

`compare` also aggregates externally attributed development-grade rows. Its
runtime/input/grader/request bindings are required declarations, not independent
authentication of an arbitrary grader. The optional Python `token_kl` function
requires full normalized distributions over the same vocabulary/positions;
top-k/truncated logits are rejected. KL is a locality diagnostic, not a model
quality metric or training objective.

Every result retains `weight_candidate_eligible: false`,
`scientific_mechanism_accepted: false` and `quality_evaluation: NOT_RUN`.
`fixed_activation_records_complete` means complete engineering/control records.
It cannot reopen either failed study or substitute for the independent Sol max
custodian. Its final runner/checker and native OpenCode/Hermes acceptance remain
separate, unimplemented/unrun gates. Native 262144 context, Q8 K/V and 28192 output
are not certified by these short CPU diagnostics.

See the [integration qualification report](experiments/unified-lab-20261109.md)
for exact test/source identities and the distinction between synthetic engineering
models and official-model research. Training and useful historical artifacts
remain preserved; cleanup is not part of this integration.

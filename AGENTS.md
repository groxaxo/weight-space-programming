# Active research objective

The active task is a **direct weight intervention on the official unmodified
Qwen3.5-4B BF16 checkpoint**, followed by matched quality evaluation. A successful
4B direct repair has not been established. Follow the method constraints in the
[pinned upstream exp8 README](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md)
and the original prompt linked there. The original `0be1a86` wrapper reconstructs
a published **0.8B** patch; it is not an edit-discovery algorithm.

The user-selected surgeon is **GPT-6.1 Sol with xhigh reasoning**. Record the
actual invocation and code/artifact identities. Do not silently substitute a
different model or reasoning effort. Surgeon access is limited to approved
practice/development evidence; final evaluation belongs to a separate custodian.
Subdito and automatic second-opinion orchestration remain disabled. Do not
invoke them or silently substitute another surgeon workflow. Explicit task
authority is required for any later change of method or delegation.

## Method boundary

- No LoRA, QLoRA, PEFT training, backward passes, autograd updates, optimizers,
  loss fitting, numerical optimizer substitutes or trained adapter merges. A
  trained adapter relabelled as a direct delta remains a trained artifact.
- Legacy training source, corpora, checkpoints, releases and measured reports are
  historical records. Preserve them. The active policy is in
  `research-policy.json`; `scripts/method_policy.py` guards legacy training,
  supervised-corpus preparation, export and serving CLI entry points. Those guards are a workflow barrier, not a sandbox for arbitrary code.
  Keep the stopped Qwen training run stopped. Do not launch a preserved script,
  supervisor or unit from an older checkout to bypass these guards. Preserve
  older source used by unrelated jobs; use the isolated active checkout for new
  direct research.
- Diagnose 4B on its own architecture and data. Never transplant 0.8B layer,
  neuron, column or flattened coordinate selections or scale their indices.
- The initial patch support is only the 32 language MLP `down_proj.weight`
  matrices with BF16 dtype and exact shape `[2560, 9216]`. Embeddings, output
  head, final norm, vision parameters and whole-tensor rescaling are forbidden.
  Extending support requires an explicit reviewed contract change.

## Evidence and custody

First reproduce a specific base-model failure under a frozen runtime, chat
template, thinking mode, seed, budget, grader and client protocol. Rule out
harness/parser/template/budget causes. Collect forward-only observations and
state a falsifiable causal hypothesis before selecting coordinates. Use no-op,
norm/support-matched random, component/layer and predeclared strength controls.
Correlation, fewer tokens and a valid numeric patch do not establish a repair.

The current `half_open_overlap` study uses `scripts/direct_causal.py` only after
an exact base replay and a token-length-matched semantic contrast. Its two frozen
stages admit six then eight forward calls: four fixed layer groups, no-op,
half-strength, two fixed random signs, off-target position and reverse transfer.
All donor artifacts, runtime scope and actual BF16 intervention norms/support
must be bound and checked. A failed or incomplete control ends that hypothesis;
do not change groups, seeds, positions, thresholds or strength after scores to
force a pass. Even a complete activation-control pass identifies an information
carrier, not defective weights or eligible patch coordinates. A separate reviewed
non-fitting derivation and practice-family regressions are required before a
candidate. Every activation-only result must retain `weights_edited: false` and
`quality_evaluation: NOT_RUN`.
The proposed next mechanism gate is in
`docs/plans/direct-mechanistic-gate-20261108.md`. It remains conditional and
`NOT_RUN` until both existing activation stages pass and its new protocol is
frozen. Never replace it with minimum-norm fitting to desired activations/logits.
The 8 November result is `STOP_controls_failed`: reverse transfer selected `>=`
but retained positive joint margin +0.8176. Preserve that verdict. It admits no
subdivision, geometry selection, weight candidate or promotion from this cycle.
A future cycle requires a new reviewed hypothesis/protocol; do not relabel or
weaken the completed gate after seeing its scores.

Practice/development data are exposed to the editor. HE9, the fourteen previous
development checks, public benchmarks already inspected and old client fixtures
are regressions, not a new untouched holdout. Never inspect independent prompts,
answers, grader internals or intermediate final scores. Keep the independent
custodian and interface separate. Freeze the candidate, code, data identities,
runtime, paired protocol, primary metric and regression/uncertainty rules before
an independent final run. Preserve rescues, regressions, failures and negative
results. A failed final evaluation requires a new research cycle and new untouched
confirmation data; it does not authorize further selection on the same holdout.
Follow `docs/direct-paired-acceptance.md`. The current practice broker is not a
final custodian or a complete paired-acceptance runner; do not claim that an
unimplemented or unrun stage passed.

## Integrity and execution

Use `configs/qwen35-4b-direct.lock.json` for official revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Fully authenticate all locked shards
and sidecars before an edit. Preserve snapshot symlink chains and their global
blob targets. Keep baseline handles read-only; never write through a shared
mapping or create a writable checkpoint hardlink.

`scripts/direct_weights.py` accepts strict numeric JSON with explicit BF16 uint16
replacement words, exact parent-lock SHA-256, shape/dtype, sorted unique indices,
selected versus actual changed counts, before/after tensor hashes and declared
magnitude/sparsity limits. It applies one tensor to a private memory copy. Do not
use pickle or execute producer artifacts. Bind hypothesis, failure, localization,
practice exposure, controls, producer code and observation records by hash. Their
declared hashes still require scientific review; schema validity does not certify
their provenance or causal interpretation. See `docs/direct-weight-editing.md`.

Admit CPU/RAM/disk/GPU work before execution. The trusted forward worker and the
editor need separate verified boundaries; neither CLI guards nor disabling an
optimizer proves that arbitrary analysis code avoided handwritten training.
Launch model workers through `ops/direct_edit54.py`, with at most 1,800 seconds
per worker and progress inspections at least every 600 seconds. Probe requests
must leave postcheck and shutdown reserve within that maximum. Do not bypass
admission by invoking the heavy worker CLI directly on the host.
Forward collection must leave the parent unchanged. Do not stop, restart,
reconfigure, pause or reroute Spark Fable Runtime, Bonsai, Moss, other jobs or the
separate evaluation to obtain resources. Queue or report an unavailable resource.
Short diagnostic contexts cannot establish the native client's 262,144 context,
Q8 K/V or 28,192 output profile. Quantized conversion and deployment require their
own pinned original-versus-edited paired experiment after BF16 evidence.

## Authorized new cycle: tool finalization

On 8 November the user authorized a new direct-weight hypothesis and independent
paired testing. Follow `docs/plans/direct-tool-finalization-cycle2-20261108.md`
and `configs/direct-tool-finalization-cycle2.protocol.json`; the previous
interval hypothesis remains closed. The new claim concerns retiring a completed
failed-test action. A native Q8 failure is not a BF16 defect until reproduced.

The initial tranche permits only two fresh full BF16 episodes, at most four
generation invocations, one per admitted CPU worker. The initial tool call must
be generated spontaneously. Continuations must authenticate the complete actual
initial episode capsule; do not supply a repaired or historical assistant call.
Use `ops/direct_tool_cycle54.py` through the shared broker, after the frozen
`ops/direct_tool_preflight54.py` template/parser check. Preserve exact system,
tool schema, result bytes, lossless argument normalization, raw special tokens,
actual parser-assigned call IDs and their tokenization invariance. Require
agreement with the independent bounded parser check. Keep all 426 parameter
postchecks and a 450-second postcheck/exit reserve inside each worker.

Both valid episodes finish regardless of the first continuation's score. Either
correct final JSON rejects the BF16 duplicate-action hypothesis; another failure
phenotype, censoring or a harness fault also prevents localization. No new layer,
feature, weight, strength or candidate selection follows a failed gate. Later
stages are conditional on their separately frozen complete controls, measured
admission and cumulative budget. The only conditional candidate is a causally
proven existing sparse write deletion; fitting and clipping fallbacks are barred.

The scientific surgeon remains `gpt-6.1-sol` with `xhigh` reasoning. The user
separately selected a Codex `gpt-6.1-sol` **max** independent custodian. Its public
contract is `docs/plans/direct-cycle2-independent-acceptance-20261108.md`. Keep
its context, private cases, answers, grading and intermediate final scores out
of candidate selection. This is separate-agent custody under the same OS user,
not an adversarial OS boundary. No hidden set is created before one eligible
candidate and all interfaces are frozen. No candidate means final paired quality
is `NOT_RUN`; the current practice broker does not become a final evaluator.

## One reviewed chat-stop correction

The first Cycle 2 four-generation batch is retained as `STOP_harness` under
producer commit `4c8cd17144b5f02e47a8a4b76dc80565873714a6`. Its generator reported
the model/tokenizer EOS union but failed to pass that union to `generate`, so
it crossed the first assistant end marker and fabricated later chat turns.
Do not trim those streams or relabel their duplicate-call prefixes as valid
trials. Do not reuse their initial-call capsules.

The new source uses
`configs/direct-tool-finalization-cycle2-chatstop.protocol.json` and the separate
stop-repair review. This is a disclosed post-fault, pre-corrected-score amendment,
not an all-data-blind original protocol. Pass the authenticated model/tokenizer
EOS union explicitly to generation. Independently require the first EOS to be
the terminal output token; internal chat controls stop before semantic parsing.
EOS at the last allowed position remains completed. Keep the original eight
harness fixes and exact all-426 postchecks.

The owner adopted Sol xhigh's single technical replay allowance: four original,
exactly one corrected tranche of four, and the unchanged five semantic controls
fit a maximum of thirteen generation invocations. Preserve the 21,600 cumulative
worker-second and 22-diagnostic ceilings and every scientific threshold. Charge
all original, failed, preflight, runtime, authentication, loading, generation,
postcheck and exit costs; no reset or free retry. The prior ledger and newly
qualified source's engineering costs must be frozen and charged before replay.
A durable same-user claim prevents accidentally starting a second correction.
Another corrected harness fault ends the cycle without a further replay.
This amendment supplies no coordinate choice, weight candidate or quality pass.

## Cycle 2 semantic-specificity gate

The corrected `f0b7bd3` two-episode batch reproduced the duplicate call with
identical greedy tokens and all 426 protected parameters unchanged. This admits
only the five exposed B/S/C/N/P semantic controls from the original plan.
`scripts/direct_tool_specificity.py` independently revalidates all four old
records against the pinned historical 13-source bank; current broker changes
must not rewrite that bank or invalidate its provenance. Its own source and the
separate host driver are additionally frozen. Never replace raw eligibility
checks with the saved gate flag.

Use `ops/direct_tool_specificity54.py` for preparation, the admitted tokenizer-only
compiler and the five one-generation workers. Stage-1 mounts contain only the
32 authenticated data files, read-only. C/N padding is a bounded tokenizer-only
first lexicographic match; no score search or wording change. Capture three
read-only positions at all 32 MLPs. Fresh B must reproduce both token sequences
and all 64 original assistant-position vectors bit for bit.

Freeze all five conditions before scores. An ordinary completed semantic failure
irreversibly rejects localization, while the remaining controls still finish.
Only infrastructure, censoring, source/integrity or budget failure aborts the
remaining tranche. Preserve the shared lease, all-426 `finally` postchecks,
one-core/10-GiB/1800-second maximum and the cumulative 13-generation/21600-second
ceiling, including all failed and engineering work. The durable same-user Stage-2
claim is evidence, not a disposable lock; never unlink it to retry.

Only a complete pass admits the separately frozen geometry/causal stages; it
does not admit a weight candidate. A failed gate admits no feature selection,
weaker cue, alternate support, fitted delta, independent final run or promotion.
The Sol max custodian receives only a later eligible immutable candidate under
its public contract. No candidate means final paired quality stays `NOT_RUN`.

## Audited worker invariants

The eight audited harness repairs are documented in
`docs/reviews/20261108-direct-harness-fixes.md`. Preserve these invariants:

- A full file SHA-256 check must remain bound to the file generation actually
  consumed. Retain inode/device/size/mtime/ctime identities, read-only descriptors
  and before/after stability checks. A changed parent requires a new full
  authentication, never a timestamp adjustment or reuse of an old proof.
- Capture input bytes and hashes before execution, preserve those exact bytes,
  and bind the producer sources captured before execution. Never hash mutable
  source paths afterward to attribute earlier results.
- Use the shared host/user worker lease before sampling resources, and hold it
  through confirmed container exit. Never unlink its inode. A dead broker PID,
  expired age or missing/unknown container state is not proof of released
  resources. Inspect only the retained owned container; do not stop other jobs.
- Check all loaded parameters in `finally`, including observation startup and
  generation interruption. Save incomplete results and propagate postcheck
  failure. Retain all partial evidence.
- Distinguish contrast completion, all requested arms, execution completion,
  scientific control success and independent model quality. Recompute cached
  gates from complete, unique, finite raw score rows. A skipped requested arm
  cannot be complete, and an execution-complete failed gate cannot be a pass.
- New host acceptance requires `evidence_schema_version: 2`: exact 426 unique
  official language parameter names and before/after hashes, full 12-file
  baseline and loaded-parent proofs, frozen source identities, tokenization,
  raw scores and numeric observations. Do not upgrade historical artifacts by
  rewriting or relabelling them. Request schema versions remain separate.
- EOS at the last allowed token is a completed generation. A length cap without
  EOS, or an unexplained short stop, remains censored. Preserve actual EOS IDs,
  token counts and stop reasons; never infer a semantic failure from a cap flag
  that contradicts the raw tokens.

These repairs do not reopen the `STOP_controls_failed` study or authorize
training, a new hypothesis, a weight candidate, model promotion or cleanup.

## Preservation and claims

Preserve all originals, intermediate artifacts, staging files, manifests,
observations and failures. No cleanup is authorized. Do not delete a path because
it is called cache/train/reports, rejected or apparently unused. Any separate
cleanup proposal needs exact ownership/dependencies, allocation, usefulness,
backup identity, restore verification and explicit deletion authorization.
Do not commit, push, publish or change provider defaults without task authority.

Run meaningful CPU contract tests and the appropriate paired model/client tests.
Report engineering integrity separately from model quality. Mark unrun steps
`NOT_RUN`; never turn reconstruction, a no-op repeat, a lower training loss or a
small development gain into a direct-edit discovery or independent quality claim.

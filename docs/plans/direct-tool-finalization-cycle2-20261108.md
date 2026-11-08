# Cycle 2: completed failed-test result and persistent action

Status at proposal freeze: `PROPOSED_NOT_RUN`. There is no BF16 reproduction,
localization, candidate or quality result in this document. A negative result is
a complete scientific outcome. The primary operator owns execution and later
publication; this surgeon writes only this plan and its adjacent proposal JSON.

## Provenance and scope

The user-selected surgeon was launched with the exact supplied tool fields
`task_name=direct_tool_surgeon`, `fork_turns=none`, `model=gpt-6.1-sol`,
`reasoning_effort=xhigh`. The returned task identity is
`/root/direct_tool_surgeon`. No opaque provider ID or backend attestation was
returned. These are actual orchestration settings, not an independently verified
claim about a provider backend. The companion JSON captures inspected source
hashes; execution must freeze its own sources again before running.

The inspected `semantic-repair-20261108-round1-paired.json` records a native Q8_0
base run in which `bounded_chunks` passed and `truthful_failure` called
`run_tests(target="parser")` again after the prescribed result. That historical
report used llama.cpp, thinking disabled, context 262144, output 28192 and Q8 K/V.
It is exposed development evidence. It is not a demonstrated BF16 defect, an
untouched holdout, or evidence for particular weights.

This is a new action-completion hypothesis, independent of interval-boundary
interpretation. The closed `half_open_overlap` verdict remains
`STOP_controls_failed`; its reverse greedy choice did not overcome its positive
joint margin (+0.8176). Do not subdivide it, change its thresholds, import its
layer/group/feature choices, or use its activation donors. The 0.8B reconstruction
also provides no discovery coordinates. Every new selection needs its own frozen
4B tool-result evidence and cannot inherit an old coordinate binding.

## Falsifiable hypothesis

After a completed `run_tests` call returns a nonzero test exit code, a small MLP
feature set may continue to write an outstanding-action signal into the next
assistant decision. This would be a failure to retire a completed action after a
negative outcome. The predicted behavior is a duplicate call, rather than merely
incorrect JSON, verbose output, or failure to call the tool initially.

The hypothesis predicts that an explicit statement that the requested execution
is complete will clear this signal, while a length-matched neutral statement will
not; successful completed runs will terminate normally; and a case that actually
requires another run will retain that ability. These predictions distinguish this
mechanism from general tool suppression, negative-result aversion, result-field
copying failure, tool parser failure and generic instruction following. Failure
of any required prediction ends this specific hypothesis. Do not rename it after
scores to keep it alive.

## Stage 0: freeze and validate the harness, without model selection

The existing `direct_forward.py` request contract admits only string contents
with system/user/assistant roles and a 256-token input cap. It cannot represent
this full tool episode faithfully. Its current code grader also accepts only
the exposed code cases. The existing interval-specific probe/causal interfaces
are not a ready-made tool evaluator. A new strict broker mode, parser contract,
tool grader and evidence verifier are implementation prerequisites. No method
gate is satisfied by merely extending a JSON schema.

Use the official authenticated tokenizer and chat template, `tools=[RUN]`, and
`enable_thinking=False`. Freeze exact system, user, tool schema, result bytes,
sampler, tokenization, EOS IDs, parser implementation, source bytes and protocol
before any BF16 score. Import the exposed `truthful_failure` fixture rather than
retyping a similar problem. Preserve raw special-token text alongside parsed
messages. A harness-generated UUID is not model-generated output: bind its
assignment and demonstrate that changing this ID does not change rendered token
IDs. Check that an actual tool result is rendered as a tool result with the
correct association; never flatten it into an assistant or user message.

Offline tests must distinguish valid tool output, valid final JSON, malformed
output, duplicate calls, missing/extra fields, wrong arguments and truncation.
Replay the same saved raw bytes through the strict parser and an independently
implemented bounded classification check. Disagreement is `STOP_harness`, not a
weight defect. Template/tokenization differences from historical Q8 behavior
remain documented runtime differences, not a claim of matched native replay.

Bind every model run with evidence schema 2: all 12 locked parent files and their
file-generation identities; exact 426 unique language parameter names and hashes
before/after; loaded-parent proof; sources captured before execution; exact input
and generated IDs, request bytes, raw output, parser source, EOS/stop/censoring
facts, observations, logs and retained container state. Recheck all 426 in
`finally`, even on startup failure or interruption. Schema validity proves
engineering integrity, not the mechanism. Never upgrade old schema-1 artifacts.

## Stage 1: two original BF16 full episodes, frozen denominator

Use the official unmodified BF16 parent, the frozen short CPU runtime, greedy
decoding, seed 20261108, thinking false, one thread, at most 512 input tokens and
96 newly generated tokens per invocation. A context that does not fit stops;
there is no silent truncation or prompt shortening. True EOS is required;
reaching the cap with EOS is complete, while a cap without EOS or unknown stop
is censored. These diagnostics do not certify the native deployment profile.

Run exactly two independent cache-reset episodes. Each episode has these steps:

1. Generate from the actual system/user/tool-schema input. The model must
   spontaneously produce exactly one valid `run_tests({"target":"parser"})`
   call, with no prerequisite result supplied in advance.
2. Preserve the actual raw and parsed assistant call, including any
   harness-assigned ID provenance. Bind the entire initial worker capsule by
   SHA-256. The continuation worker independently verifies its schema-2 evidence,
   parent identity, source identity and exact 426 map before using the call.
3. Append that actual assistant message and a correctly associated tool message
   whose exact content is `{"exit_code":1,"passed":4,"failed":2}`. Generate the
   next assistant message from this complete conversation with fresh model/cache
   state. No repaired or synthetic assistant call may be substituted.

One generation invocation per admitted worker is the conservative initial plan:
four invocations for two valid episodes. A generation invocation contains a
prefill and potentially many cached token forwards; do not call this four
forward passes. Complete both valid episodes regardless of whether the first
continuation repeats the call. This fixed denominator is frozen before scores.
An invalid initial call fails admission and ends that episode; it cannot be
continued by manufacturing the expected call.

The reproduction gate passes only if both uncensored, attributed full episodes
spontaneously make the expected initial call and then issue the same valid
`run_tests(target="parser")` call again. Record the duplicate as a completed
undesired action even though no real tests are executed. Correct final JSON in
either episode means `STOP_BF16_target_not_reproduced`; another completed failure
means `STOP_different_phenotype`; malformed/censored/unknown results mean
`STOP_inconclusive` or `STOP_harness` according to their actual cause. The gate
fails irreversibly upon any such result, but the fixed second valid episode still
finishes. Do not cue, localize or select weights after a failed gate.

A historical assistant-call prefix may be replayed only as a separately labelled
conditional diagnostic. Two repetitions of that prefix do not replace the full
episodes or demonstrate that BF16 produced the prerequisite call.

## Stage 2: five fixed development continuations and a new observation

Run only after Stage 1 passes. Before their scores, freeze five conditional
post-result conversations derived from one authenticated Stage-1 episode:

| ID | Sole planned change | Required completed behavior |
| --- | --- | --- |
| B | Exact failed-result baseline, fresh replay | Same duplicate call |
| S | Result becomes `{"exit_code":0,"passed":6,"failed":0}` | Final JSON `{"exit_code":0,"failed":0}` |
| C | Failed result; append the fixed completion cue to the original user request | Final JSON `{"exit_code":1,"failed":2}` |
| N | Failed result; append a fixed neutral, tokenizer-length-matched cue | Same duplicate call |
| P | User explicitly requests two runs, reporting the second; exactly one result is present | Valid second parser-test call |

The completion cue is: `The requested test execution is complete; report its
observed result now.` The neutral cue is: `The requested test execution concerns
the parser; its recorded details appear here.` Before any model scores, match
C/N token counts and the decision position using a deterministic tokenizer-only
padding rule with repetitions of the word `recorded` followed by a period. Search
at most 32 padding counts per cue in lexicographic count order, choosing the
first exact match. If no semantically neutral exact match fits the 512-token
limit, stop this contrast. The cue texts/counts may not be changed using model
scores. P's frozen instruction is: `Run the unit tests for parser twice. After
the first result, run the same tests once more. After the second result, return
only a JSON object with keys exit_code and failed copied from that second
result.` Preserve the original system and tool schema for all five cases.

These cases are exposed development controls, not independent evaluation. S
passing is part of this narrow nonzero-outcome hypothesis; if S also repeats,
stop rather than switch to a general finalization claim. Any required behavior
failure, incomplete case or template mismatch is `STOP_semantic_specificity` or
the corresponding integrity stop. Preserve full generations, not just token
margins. A cue that only suppresses all tools cannot pass P.

Collect all 32 down-projection inputs/outputs at the last assistant-prompt token
in these new contexts, with separate explicit positions for the last tool-result
token and prior-call token. Additional gate/up observations must be declared
before collection, checked against the actual down-projection input, and bound
to exact tokens. Stage 1 observations are not permission to choose coordinates.

## Stage 3: bounded localization, not a weight candidate

Only after Stage 2 passes, freeze the new localization request. Do not call the
old interval probe with renamed cases. Use all 32 newly observed layers once,
without old groups, priors or subdivisions. For layer l define the persistence
contrast `d_l = y_N(l) - y_C(l)` at the exact shared decision position. Reject a
zero, nonfinite or nondeterministic contrast. Examine exact column contributions
`(g_N[j]-g_C[j]) * dot(W[:,j], d_l) / dot(d_l,d_l)`, verifying numerically that
their sum reconstructs the observed projection up to recorded BF16 matmul error.
This is an observational ranking, not proof of a causal feature.

Choose at most one layer by a frozen dominance rule: its positive reconstructed
contrast mass, before division by `dot(d_l,d_l)`, must exceed half the sum of
those unnormalized positive masses over all layers; a tie
or no dominant layer stops the single-layer track. At that layer take the
smallest descending-contribution set J, stable index tie-break, explaining at
least 80% of its positive contrast, with at most eight features. If more than
eight are needed, stop. Do not expand support, try the runner-up, subdivide, or
import any previous study's features. A different observed mechanism requires
a future protocol, not a silent replacement.

Freeze complete tool-call versus exact-result JSON branch token sequences and
their first divergence before activation tests. They must round-trip to the
actual tool format and exact shared assistant prefix. Use joint log probability
over each complete fixed branch without length normalization, plus the greedy
first divergent token. A branch margin is an auxiliary conditional action
diagnostic; it is not executable quality or a generated success. If canonical
branches cannot express the observed decision faithfully, stop this diagnostic.

At most 12 newly executed diagnostic arms are permitted: neutral B replay, C
replay, full contrast removal from B, half removal, reverse addition to C, two
fixed hash-seeded norm/support-matched random directions, the same perturbation
at a predeclared different layer, an off-target token position, S with removal,
P with removal, and the corresponding untouched S/P action-score baseline
record (one declared batched invocation, accounted for as two examples).
Freeze the other layer as `(l+1) mod 32`; freeze the off-target position as the
last tool-result token. Never choose either by scores. Account for examples and
model calls separately, including each branch row in a scoring batch.

Require full removal to select the valid final-JSON branch and positive joint
margin, reverse addition to select the duplicate-call branch and negative joint
margin, and a strictly intermediate half-removal margin between B and full.
The reverse joint-margin sign is mandatory even when greedy tokens agree.
Random, other-layer and off-target controls must retain B's duplicate action and
negative margin; S and P must retain their respective valid action and margin
sign. Require numeric no-op identity. Match actual applied BF16 norm/support,
not requested FP32 values; random controls outside a predeclared 5% norm
tolerance are invalid and stop rather than get resampled. All arms must complete
and agree with the raw score arithmetic. One failed control is
`STOP_activation_controls`; no altered seeds, strengths, thresholds or groups.

Even a pass identifies an information carrier. It does not justify editing W.

## Stage 4: sparse read/write causality, or STOP

Require independently checked gate/up decomposition
`g = silu(W_gate h) * (W_up h)`, exact actual down-projection input agreement and
the selected features' signed existing contributions `g[j] * W[:,j]`. Their
read activation must distinguish B/N from C/S and from the legitimate pending
case P. Freeze this numerical read-specificity check: for every selected
feature, B/N inputs have the same nonzero sign and their smaller absolute
magnitude is at least four times the largest absolute input among C, S, P and
the initial no-result state. A zero control maximum still requires nonzero B/N.
This is necessary observational specificity, not sufficient causal evidence.
Merely carrying the word `failed`, JSON digits, generic tool syntax or
all tool use is insufficient. Record alternative explanations explicitly.

Freeze at most ten further diagnostic arms: untouched replay, J input-feature
ablation, restoring J's original inputs to that ablation, reverse addition of
the exact removed existing write to C, two fixed support/norm-matched random
feature-set controls, J ablation at the off-target position, and J ablation in
S, P and the initial no-result tool-call state. Selection of random sets uses
fixed SHA-256 seeds; no score-based retries. Require ablation rescue and signed
reverse failure as above, restoration/no-op equality, no rescue by random or
off-target controls, and preserved correct initial/pending tool use and S
reporting. Require J to explain at least 80% of the signed causal removal seen
in Stage 3 without introducing an unrelated broad write. Whole-output rescue
with weak, diffuse, nonspecific or noncausal feature evidence is
`STOP_sparse_mechanism`, not a reason to fit a matrix.

## Conditional, single candidate derivation

There is no general non-fitting weight repair implied by an activation donor.
This protocol admits only one narrow algebraic candidate if every preceding
gate proves an existing sparse maladaptive write: remove that already isolated
write by setting the complete authenticated down-projection columns J to BF16
positive zero. No other entries or tensors change. This is a causal component
deletion tested before weight editing, not clipping large weights or choosing
columns by magnitude. It does not claim the checkpoint was historically corrupt.
If the feature evidence does not justify complete deletion, STOP; there is no
projection, minimum-norm solve, pseudoinverse, rank-one target matching, desired
activation/logit fitting, sign/scale search or partial-column clipping fallback.

Strength is exactly one, fixed before effects; half strength was only a causal
control. At most one layer and eight full columns means at most 20,480 selected
words. Before deriving words, check the actual removed write and all existing
lock ceilings, including absolute delta 0.125, relative L2 0.05 and fraction
0.01. If complete deletion violates any ceiling, stop; do not clip, rescale,
reduce J after scores or raise bounds. Recompute the BF16 intervention itself,
including zero signs, actual changed count, complete before/after hashes and
every protected parameter. Only the trusted validator applies explicit sorted
uint16 words to a private verified copy. No shared writable mapping or new full
checkpoint is needed for this gate.

The candidate then freezes without tuning. Before exposed weight-effect
evaluation, separately freeze no-op, half-deletion and two hash-seeded random
column-deletion controls, including a different-layer component control. Use
the same number of selected columns and require actual changed support and
BF16 delta norms within the predeclared 5% tolerance; a fixed random control
that cannot meet this contract is a stop, not permission to search seeds or
rescale it. Half deletion is a control only and may never replace a failed full
candidate. This later weight-control tranche requires separate complete resource
admission and its frozen paired protocol; it is not hidden inside the discovery
budget or presumed implemented by the practice broker.

The frozen candidate still needs original-versus-edited
BF16 full episodes, all exposed regression families including base-solved
`bounded_chunks`, truthful reporting, valid initial/repeated/dependent calls,
and nonzero-error recovery. Preserve rescues and regressions with executable
grading. Any regression rejects this candidate. Candidate derivation is not
quality acceptance and cannot silently authorize more selection.

## Resource, custody and claim boundaries

All heavy work goes through the shared `ops/direct_edit54.py` lease and admitted
CPU container: one core, 10 GiB, no GPU, at most 1800 seconds per worker, progress
inspection at least every 600 seconds. Reserve at least 450 seconds inside each
worker for postcheck and confirmed exit. Stop before another invocation if the
measured worst-case cost plus this reserve will not fit. Current approximately
380-second cost for a 375-token diagnostic forward is a planning observation,
not a guaranteed generation time. Each scoring branch/example counts. The
initial authorization window is at most four generation workers (7200 seconds
total worker wall); later stages require their own measured admission and
frozen requests. Cap the entire new discovery cycle at 21,600 cumulative worker
seconds, nine generation invocations and 22 additional diagnostic invocations,
whichever limit is reached first. The batched S/P baseline consumes its actual
two-example cost; no batching claim can evade the cumulative cap. If the full
planned tranche does not fit, mark it `NOT_RUN` and stop without a candidate.
These ceilings are not instructions to exhaust the budget.

Do not preempt, reroute or restart occupied services or other jobs. Insufficient
RAM/disk/time means STOP/NOT_RUN. Preserve every artifact, partial run, model,
VM and service; cleanup is not authorized. This surgeon launches no workers and
does not commit or push. The primary operator owns reviewed source integration.

No independently controlled holdout custodian has been identified. Do not create,
inspect or claim one here. Development evidence belongs to the editor. Final
custody requires a separately controlled interface, opaque task identities and
frozen task count/families, paired seeds/order, primary executable metric,
minimum gain, regression limits, censoring rules and uncertainty calculation
before either final arm. Until then final quality is `NOT_RUN`. Failed final
acceptance rejects the frozen candidate and requires a new research cycle and
new untouched confirmation data.

Finally, preserve the separate deployment gate: authenticated original/edited
conversion and quantization, matched Q8 behavior, native OpenCode/Hermes context
262144, Q8 K/V and output 28192, actual attributed client tests and fresh-VM proof
where required. A 512-token CPU probe certifies none of those properties.

# Cycle 2 stop-boundary repair review

Review status: `PROSPECTIVE_REPAIR_RECOMMENDED_NOT_EXECUTED_BY_SURGEON`.
Surgeon identity remains `/root/direct_tool_surgeon`, launched as
`gpt-6.1-sol` with `xhigh` reasoning, `fork_turns=none`. The actual supplied
launch settings were recorded in the original proposal; provider backend
attestation is unavailable. This review changes neither the original plan nor
the original proposal. No source, worker, service, weight or independent case is
changed or executed by this surgeon.

## Finding and scientific verdict

The primary operator reports original Stage 1 execution from immutable commit
`4c8cd17144b5f02e47a8a4b76dc80565873714a6`. This review inspected the local
source snapshot identified in the adjacent JSON; local Git metadata is absent,
so that snapshot is not independently asserted to equal the reported commit.
Original worker source manifests must establish the exact correspondence.

The first reported initial generation made a valid call in 28 tokens. The first
reported continuation reached EOS after 83 tokens, but its raw stream contains
a complete duplicate tool call, an assistant `<|im_end|>`, a generated user/tool
response, another assistant turn and a fenced JSON answer before
`<|endoftext|>`. These are exposed development observations. They reveal the
harness fault; they are not holdout evidence and are not an admissible result
for the intended single assistant turn.

The inspected `direct_forward.observe` passes greedy decoding, token budget and
padding identity to `model.generate` but does not explicitly pass its chat-stop
union. `completion_status` independently collects model/tokenizer EOS IDs and
checks only the last generated token. `direct_evidence.verify_observation`
similarly recomputes last-token completion from the row's EOS list, without
requiring the first stop to be terminal. Thus reporting a broader stop list does
not prove generation actually used it. The existing
`serve_coder.chat_stop_tokens` passes the model/tokenizer union to generation;
its comment describes the checkpoint/assistant-turn EOS distinction. This is
relevant local evidence for the fault, not permission to start a serving process.

The original batch and its exact `STOP_harness` verdict must remain immutable.
Finishing its two fixed valid episodes unchanged, as already chosen by the
operator, preserves the denominator. Neither a duplicate-call prefix nor the
later JSON may be extracted as a completed scientific trial. Do not trim at
`<|im_end|>`, relabel historical output, regrade old artifacts under a new stop
policy, splice old prefixes into new episodes, or claim reproduction/localization
from the old stream. The fault makes the original instrument inconclusive about
the target BF16 phenotype. It does not establish a repair, falsify the hypothesis
or prove the hypothesis true. Activation admission remains false.

## Required prospective generation and verification contract

The repair is a runtime correctness correction, not a method/weight change.
Freeze a new stop policy and new protocol/source identity before any corrected
BF16 generation. Keep model, official parent, fixture, system/user messages,
tools, thinking mode, greedy sampler, seed, input/output budget, parser semantic
contract, episode count and later scientific controls unchanged.

1. Before generation, derive a sorted unique union from the authenticated
   model-derived generation EOS setting, tokenizer EOS setting, and explicit
   authenticated token identities for `<|im_end|>` and `<|endoftext|>`. Validate
   exact integer types, vocabulary membership and round-trip token identities;
   neither missing literals nor an unknown-token fallback is acceptable. Do not
   merely hardcode IDs remembered from another runtime. The historical native
   report's `[248044, 248046]` is a comparison observation, not authentication.
2. Trace the resolved model generation configuration to locked inputs. The
   current official lock includes config/tokenizer/template files but does not
   list a separate `generation_config.json`. If such an additional file is
   actually consumed, stop and authenticate that dependency through a reviewed
   expanded input contract before using it. Do not silently read an unbound
   neighboring snapshot file. A generation configuration derived exclusively
   from the authenticated model config must say so explicitly.
3. Pass the exact frozen union as `eos_token_id` to the actual `generate` call.
   Record both intended and effective arguments before execution. Completion
   reporting must consume this same effective union; it must not reconstruct a
   potentially different set afterward. Keep true EOS-at-cap completion and
   cap-without-EOS censoring unchanged.
4. On the returned generated-token array alone, independently find the first
   token in the union. A completed single-row invocation requires that token at
   exactly `len(output_ids)-1`, with no tail whatsoever. Two consecutive stop
   tokens are not a single valid terminal event. A stop before the final token
   is `STOP_harness`; no stop at the cap is censored; an unexplained short stop
   without any authenticated stop is inconclusive. Preserve the original full
   array and the index, token identity and length of any invalid tail.
5. Before semantic parsing, reject assistant-generated role framing such as
   `<|im_start|>` anywhere, and reject internal `<|im_end|>` or
   `<|endoftext|>` markers. This fixed task does not require those markers as
   literal response data. The check covers both actual special-token IDs and
   decoded role/control markers so a textual imitation cannot bypass it. A
   generated `<tool_response>`/user turn is not an authorized actual tool result.
   Retain support for the already allowed tool-call XML and thinking framing;
   do not broaden semantic parsing or fenced-JSON acceptance during this fix.
6. The independent checker must verify first-stop terminality from exact token
   IDs and authenticated policy, then validate the bounded raw response. The
   primary parser and independent checker may remove exactly the verified final
   stop for syntax analysis, but cannot remove an arbitrary suffix. Keep the
   full raw string available. If either sees an internal control token, classify
   it as a harness stop even when the two semantic parsers would otherwise
   agree. Parser agreement is not a substitute for the stop boundary.

Use a small pure stop-policy helper if needed; record that helper in the frozen
producer list. Do not import or invoke a legacy serving/training entry point to
obtain its utility function. The engineering implementation belongs to the
primary operator and must keep the existing 426-parameter `finally` checks and
baseline generation-stability checks on every path.

## Required source boundary and retained records

The original boundary is the original plan/proposal/protocol hashes and the
reported `4c8cd17144b5f02e47a8a4b76dc80565873714a6` worker source commit, with
actual per-worker source manifests. Preserve all four original invocation
records, request bytes, initial capsules, generation arrays/journals, parser
results, preflights, logs, timings, errors, loaded-parent proofs, full parameter
postchecks, broker state and retained containers. The old protocol's
`reviewed_parent_commit` is not interchangeable with its actual producer commit.

The corrected boundary must be a new immutable commit/source bundle, new
protocol revision and new SHA-256, new preflight artifact, and a fresh result
directory. Its exact future commit/hash is not yet known and must not be invented
in this review. Keep the original protocol bytes under their original identity;
prefer a distinct versioned file for the corrected protocol. Add explicit
`protocol_revision`, `supersedes_for_future_execution_only`, `repair_reason`,
`previous_attempt_artifact_bindings`, and the budget amendment to its manifest.
Request schema versions and evidence schema 2 need not be relabelled as protocol
revision numbers.

Freeze before execution: source bytes/hashes for the forward worker, stop helper,
tool observer, independent checker, evidence verifier, broker, batch runner,
preflight and imported parser/fixture/sampler dependencies; code commit and bundle
hash; lock, original plan/proposal, new review and new protocol hashes; dependency
versions, container image, resource settings and generation arguments. The
effective-stop proof records model/tokenizer EOS inputs, their authenticated
source file hashes, explicit literal token IDs/round-trips, the final union and
the exact argument passed to generation. The generation record binds output IDs,
raw/decoded output, first-stop index/ID, token count, EOS/censoring facts and that
proof. The independent verifier recomputes the policy from bound inputs rather
than trusting the reported union.

Retain full authentication of all 12 locked files and their actual consumed file
generations, exact 426 unique original language parameter hashes before and
after, official dtype restorations and source identities captured before work.
Any needed new config dependency must be separately admitted and authenticated,
never inferred from a filename. Binding records after a run cannot retroactively
attribute mutable source paths to that run.

Do not consume an original-attempt initial capsule in the corrected continuation.
Run both initial calls again. Corrected continuation capsules must verify the
new protocol hash, new frozen producer identities, exact original parent and
426 map anew. No old/new episode mixing is permitted.

## Required engineering tests before corrected model scores

Use synthetic token arrays/stub generation, not extra official-model probes, to
test explicit EOS argument transmission and the independent verifier:

- Model EOS and tokenizer EOS differ; both stop IDs and both explicit literals
  appear in the passed union. Invalid booleans, missing/unknown literals,
  out-of-range IDs and unbound generation-config input fail closed.
- Either legitimate final EOS is accepted, including EOS in the last allowed
  position. No EOS at the cap is censored. Short no-EOS output is inconclusive.
- An internal `im_end` followed by fabricated user/assistant text and later EOS
  is rejected from raw IDs before semantic parsing. Internal endoftext, adjacent
  final EOS tokens, `im_start`, and textual control-marker imitation are rejected.
- The saved 83-token failure, if available to the primary operator with its exact
  original token IDs, is a regression fixture that remains a harness failure;
  it is not trimmed into a passing or reproduced model trial.
- Existing valid tool-call formats, actual UUID attribution, input rendering,
  parser disagreement rules, capsule tamper checks, true EOS-at-cap behavior and
  all-426 postchecks on interruption remain intact.

Record exact CPU test commands/results and new preflight hashes. A template-only
preflight cannot verify an EOS argument that was never passed to generation;
the stub must assert the actual forwarded keyword argument, and the real worker
must save its effective argument proof before calling the model.

## Prospective replay and budget recommendation

A new corrected full-episode replay is scientifically valid after a demonstrated
instrument fault. This is a **post-fault, pre-corrected-score amendment**; do not
claim that the whole study remains pristine pre-score preregistration. The old
raw output was observed, including its duplicate-call prefix. Disclose it. The
technical rule is justified by the assistant-turn boundary and by the existing
native serving path, not selected to rescue a desired score.

Recommend one documented technical replay of the same four generation
invocations, with a cumulative generation ceiling of **13** for this cycle:
original four + corrected four + the original fixed five Stage-2 continuations.
This is a resource-count amendment, not permission to change scientific gates.
The primary task owner can adopt it within the user's broad research instruction;
this review does not itself launch or implement it. If the owner retains the
original nine-invocation ceiling, the corrected four may run but Stage 2 cannot
fit its full five-case tranche; stop before Stage 2 and report `NOT_RUN`, rather
than run a selective subset.

Keep every other ceiling unchanged: **21600 cumulative worker-seconds**, at most
22 further diagnostic invocations, one core/10 GiB/no GPU, at most 1800 seconds
per worker, at least 450 seconds reserved for postcheck/exit and progress
inspection no less frequently than 600 seconds. The original-attempt four
workers, any failed/partial worker, original and corrected preflight workers,
parent authentication, loading, generation, final hashing and confirmed exit all
remain charged in the cumulative worker ledger. Count actual elapsed worker wall
time, including overhead. Keep broker wall time separately if it differs; do
not double-count overlapping categories or count queue idle time as a fictitious
model invocation. Never reset the ledger or start a new six-hour clock at repair.
Unknown original worker cost requires reconciliation before further admission.

The original four-invocation and corrected four-invocation batches each retain
their own maximum 7200-second window; cumulative admission can be stricter.
Before each tranche, reserve enough measured capacity for the entire fixed
tranche plus postchecks. If time remaining cannot accommodate all five semantic
controls or all required later controls, stop that tranche `NOT_RUN`. There is
no instruction to exhaust a budget. No services/jobs may be preempted, restarted,
rerouted or deleted to make it fit.

This recommendation permits exactly one four-invocation technical replay, not a
repeat-until-success policy. Any further stop-policy/parser/harness fault in the
corrected instrument ends this cycle `STOP_harness`; preserve it and do not
raise counts again. Two independent fresh episodes still run to their fixed
denominator when initial calls are valid, regardless of the first continuation.
Invalid initial calls are not replaced. Correct JSON in either valid corrected
episode means `STOP_BF16_target_not_reproduced`; another completed phenotype
means `STOP_different_phenotype`; censoring/unknown stops remain inconclusive.
Only both valid identical duplicate calls admit the unchanged semantic controls.

The correction creates a new evidence branch; it does not convert the original
failed branch into a pass. An activation-only success remains insufficient for
weights. No selected coordinates, stronger perturbations, new cue search, gate
relaxation, fitting, optimizer or weight candidate is authorized by this review.
The previous interval study stays closed. The supplied independent custodian is
Sol/max with a public interface only and no hidden cases created, according to
the primary operator; this review neither inspects nor creates final material,
and does not assert an adversarial custody boundary. Final quality, candidate
derivation, native Q8/262144-context/28192-output client acceptance and deployment
all remain `NOT_RUN`.

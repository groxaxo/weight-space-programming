# Conditional Stage 2 implementation and integration contract

Implementation status: `CONDITIONAL_NOT_RUN`. This change supplies a strict
compiler, one-invocation CPU worker and evidence verifier in
`scripts/direct_tool_specificity.py`, plus synthetic CPU contract tests. It does
not integrate the host broker/controller, run official tokenizer/model jobs,
create final cases, select features or derive weights. No corrected Stage-1
model score was available when the control and count policy was chosen.

The surgeon remains `/root/direct_tool_surgeon`, supplied launch model
`gpt-6.1-sol`, reasoning `xhigh`, full-context fork disabled. Provider-side model
attestation is unavailable. Work is confined to the separate local Stage-2
snapshot and the three assigned files. The immutable active `.54` corrected
source and all previous artifacts are untouched.

## Scientific protocol preserved

Run all five controls in fixed B/S/C/N/P order after full Stage-1 eligibility.
An ordinary completed semantic failure irreversibly rejects the gate but does
not shorten its denominator: finish all five completed controls. Stop remaining
invocations only for infrastructure, source/integrity, censoring or resource
budget failure, retaining explicit `NOT_RUN` rows. There is no adaptive prompt,
criterion, strength or support change.

- B: exact original failed-result continuation; require the same parser call.
- S: only result bytes become `{"exit_code":0,"passed":6,"failed":0}`; require
  final JSON `{"exit_code":0,"failed":0}`.
- C: failed result and exact original completion cue; require final JSON
  `{"exit_code":1,"failed":2}`.
- N: failed result and exact original neutral cue, tokenizer-length matched to C;
  require the same parser call.
- P: exact original twice-run instruction, with one actual completed result;
  require the legitimate second parser call.

The call rule is identical for B/N/P: exactly one `run_tests` call with the exact
string argument `target="parser"`, no additional argument/call and parsed
assistant `content is None`. Accompanying narration does not pass. S/C require
the exact two integer keys and values, no booleans, extras, duplicate JSON keys
or tool calls. This code does not expand fenced-JSON acceptance. Strict primary
and independent parsing must agree; internal chat controls, fabricated turns,
censoring and parser faults cannot become passes.

The exact cue constants and P instruction come from the original scientific
plan. Padding counts are 0 through 31, at most 32 options per cue, using only
repetitions of `recorded` followed by a period. Cache the 32 C and 32 N rendered
tokenizations and take the first lexicographic `(C_count,N_count)` with identical
full input length at most 512. No scores enter this search. Failure to match
stops the contrast. No truncation or alternative wording is allowed.

## Historical source boundary

The Stage-1 bank is pinned to immutable commit
`f0b7bd3674e3516611af786aca7ffc27fdbc7f57`, protocol SHA-256
`d44e668c125c3e3fb28ea756dcde83f0facbc7ec1d8a5e9fe589944152edec15`
and qualification-ledger SHA-256
`c84c57a09f59bc0d25b7ea66f4f5d950dc5f567d2571001ea1db3eb8782d6893`.
The immutable 13 expected producer hashes are literal reviewed constants in the
new module and checked against that ledger. Incoming reports cannot nominate
their own trusted producer bank. Old v1/4c8/other-protocol records are rejected.

`load_stage1(stage1_root, lock)` expects the retained batch layout
`episode-{1,2}-{initial,continuation}/worker-output/`. It checks all four records
with `evidence.verify_common(..., source_hashes=STAGE1_SOURCES)` and
`evidence.verify_observation`. Initial records additionally use
`direct_tools.verify_tool_evidence` in INITIAL phase, which has no recursive
capsule comparison. Continuation proofs are checked against those independently
verified actual initials, their full fixed artifact maps, tokenization and raw
classification. The full four-record reproduction gate is recomputed, including
exact paired input/output token replay and protected-parent identity. A saved
pass flag or a historical assistant prefix cannot admit Stage 2.

The manifest binds 32 fixed Stage-1 artifact paths/hashes. New current broker or
evidence hashes may differ from the historical bank. The original semantic
helpers (`direct_tools`, `coder_common`, `evaluate_coder`) must still match their
reviewed bank identities; silently changing their interpretation is rejected.
`evidence.PRODUCER_FILES` remains the existing 13 paths. New code is separately
bound as `specificity_producer_sha256`, in addition to current common producer
hashes. No monkey-patching of the historical verifier's current-source global is
used to admit old records.

## Primary-owned integration sequence

After the corrected gate passes, integrate this module into a new task-owned
source bundle; never change the running corrected source. The primary owns
broker routing, container mounts, full-tranche budget admission and cost ledger.

1. Freeze all current common sources and this new module. Start a fresh Python
   process so import-time source hashes match that frozen bundle. The common
   producer list stays 13; separately capture this module's exact source bytes
   and SHA-256. Mount source, protocol, conditions and the complete retained
   Stage-1 root read-only; only the fresh output directory is writable.
2. Call `stage1 = load_stage1(stage1_root, load_lock())` before loading a real
   tokenizer. This independently verifies eligibility and returns the manifest
   hash. Call `protocol_document(stage1, lock)` to obtain the exact strict Stage-2
   protocol shape; save and pin those bytes. This includes the unchanged parent,
   fixture, profile, all five controls, current 13 source hashes, new module hash,
   old protocol/commit, manifest and the fixed count/resource policy.
3. In an admitted, charged tokenizer-only compiler/preflight, authenticate the
   official parent inputs and instantiate the official local tokenizer. Call
   `compile_conditions(tokenizer, stage1_root, protocol_path,
   expected_protocol_sha256, lock)`. Eligibility is reverified before this API
   performs any tokenizer operation. Save/pin the returned strict JSON bytes.
   It contains all five complete prompt proofs, token IDs/offsets, first-match
   padding record, manifest and parent parameter map. The compiler does not
   write files or launch any model itself.
4. Admit the complete five-generation tranche against the shared lease, all
   original/qualification/corrected/compile costs and the existing 13-generation
   and 21600-worker-second cumulative limits. No count or wall clock resets.
   Before scores, freeze one strict request per control with the keys shown
   below. Use one new admitted container and one generation invocation per
   request, with fresh model/cache state.
5. Invoke the worker CLI below only through that admitted host route. Then call
   `verify_specificity_evidence(root, report, request, protocol,
   protocol_sha256, conditions, stage1, lock)` with independently host-pinned
   protocol/conditions and freshly checked Stage-1 evidence. It checks common
   schema 2, the separate new source identity, saved inputs, effective generation
   arguments, all three numeric archives and raw classification. Collect
   `{control, worker: report, analysis: returned_analysis}` records and call
   `specificity_gate(records)`. The gate recomputes raw classifications and
   parameter preservation; a cached boolean cannot substitute.

Worker CLI (paths are broker-mounted read-only inputs; this is documentation,
not an instruction to bypass admission):

```text
python -u scripts/direct_tool_specificity.py \
  --request /request.json --conditions /conditions.json \
  --stage1 /stage1 --protocol /specificity-protocol.json \
  --protocol-sha256 <host-pinned-exact-file-sha256> --output /out/worker.json
```

`/out` must be a fresh task-owned directory, with no existing result or input
snapshots. The worker refuses replacement and saves exact request, conditions
and protocol bytes before model execution. The CLI does not create a broker,
GPU process, network route, arbitrary hook or generated-code execution path.

The request's exact keys are `schema_version:1`, `role:"practice"`,
`question:QUESTION`, `control` (one of B/S/C/N/P), `lock_sha256`,
`protocol_sha256`, `conditions_sha256`, `stage1_manifest_sha256`, and
`generation:PROFILE`. Unknown fields and altered profiles are rejected.
`protocol_document` is the reference constructor for the protocol's exact keys;
the host must pin its hash through the separate CLI argument. The worker also
recompiles all prepared conditions using the authenticated runtime tokenizer and
requires exact equality before generation; prepared tokens cannot inject a
different prompt or choose a different observation location.

## Observation and runtime contract

The shared authentic loader supplies the official original BF16/FP32 language
parameters. Require all 426 before hashes to equal the admitted Stage-1 parent.
Only read-only `down_proj` hooks are installed; they return no replacement.
Capture all 32 input/output vectors at each of three positions in the first
prefill, then ignore cached token forwards:

- `assistant_prompt`: the final actual model-input token.
- `tool_result_data`: the BPE token covering the final character of the actual
  result JSON inside the tool-response data span.
- `prior_call_data`: the BPE token covering the final character of `parser` in
  the actual prior assistant tool call. Tool syntax examples in the system
  preamble do not qualify.

Each prompt proof records rendered text, full input IDs, full tokenizer offset
mapping, exact selected character/data span, token position/ID and selected
token offsets. A token may absorb adjacent whitespace, but cannot cross into a
different non-whitespace field/control marker. Rendered-template encoding,
template tokenization and fast-tokenizer offset IDs must agree. IDs assigned by
the parser remain explicitly harness-originated; changing the ID must not change
rendered input tokens.

There are three separate uncompressed bounded NPZ archives, exactly 64 numeric
BF16 uint16 arrays each: 32 inputs of width 9216 and 32 outputs of width 2560.
The existing bounded NPZ reader rejects object data, unsafe archives and nonfinite
words. The main observation aliases the assistant-prompt archive. Fresh B must
match the admitted Stage-1 continuation's input/output IDs and all 64 original
assistant-position vectors bit for bit. A mismatch is an integrity/replay stop,
not permission to tolerate a different baseline.

The worker uses the exact 512-input/96-output greedy profile, seed 20261108,
thinking false, one core/10 GiB/no GPU. It explicitly passes the `[248044,248046]`
authenticated model/tokenizer chat-stop union after checking it against the
protocol; completion retains the first-stop-terminal and EOS-at-cap rules.
The active alarm is 1250 seconds. `finally` re-arms a 450-second bounded
postcheck/exit reserve, so the worker remains below the 1800-second broker cap.
All 426 after hashes and baseline file-generation stability checks run even after
prompt preparation, observation or generation failure. Partial arrays, token
journals, errors and postcheck results are retained; no partial run is a pass.

## Validation and remaining gates

Synthetic CPU tests cover the historical/current source separation, revalidation
of four episodes, eligibility before tokenization, fixed cues and lexicographic
padding, exact B input replay, offset proofs including preamble examples, strict
messages/profiles, control-specific parsing, hallucinated turns/censoring,
complete denominator and irreversible failure, all-426 postcheck after an
interrupted preparation, and host binding of all three archives/new sources.
No test loads Qwen weights, a real tokenizer, Torch, a GPU or a final case.

Executed locally with `PYTHONDONTWRITEBYTECODE=1` and the supplied task venv:

- `python -m unittest discover -s tests -p test_direct_tool_specificity.py -v`:
  15 tests passed.
- `python -m unittest discover -s tests -p 'test_direct_*.py' -q`:
  176 tests passed, 11 skipped, 1.678 seconds.
- Full discovery: 431 tests, 4 failures, 37 errors, 36 skipped. Every failure/error
  was inspected: Linux `renameat2` publication and `/proc` assumptions, or local
  HTTP socket binding rejected by this Mac sandbox. The subsequent
  `verify-output` failure follows the unavailable Linux publication. No new
  specificity test failed. The full suite is not reported as passed, and no
  environment-dependent checks were weakened or repaired in unrelated files.

The local environment lacks Torch. Actual installed-Torch hook execution,
authenticated Qwen fast-tokenizer offset behavior, complete worker CLI/broker
routing and a real 1800-second admitted runtime still need primary-owned
qualification after eligibility. A failure to fit the runtime or obtain exact
offset/length/replay proof is `NOT_RUN`/STOP; do not weaken the contract. Gate/up
observations are deliberately deferred to their later separately frozen gate.
Scientific Stage 2, localization, weights, independent quality and native
deployment all remain `NOT_RUN` until their real gates run.

# F09: reported EOS union was not passed to generation

The new full tool-episode experiment exposed a further execution discrepancy
under producer `4c8cd17144b5f02e47a8a4b76dc80565873714a6`. This is a stopping-profile
repair, not a Qwen weight repair. The original four-generation batch completed
but its scientific gate is **`STOP_harness`**, with no localization or candidate
admitted. The original source, requests, outputs, observations and verdict remain
unchanged in the original `.54` worktree.

## Cause and preserved evidence

The official model configuration uses EOS `248044` (`<|endoftext|>`), while the
authenticated tokenizer uses `248046` (`<|im_end|>`). The native serving helper
already supplies their union. The shared CPU observer collected that union for
completion metadata but omitted `eos_token_id` from `model.generate`. Generation
therefore used the checkpoint default and continued past the assistant's end.

| Episode | Phase | Input tokens | Output tokens | First EOS index | Last output index | Original classification |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Initial | 317 | 28 | 25 | 27 | expected_initial_call |
| 1 | Continuation | 374 | 83 | 25 | 82 | STOP_harness |
| 2 | Initial | 317 | 28 | 25 | 27 | expected_initial_call |
| 2 | Continuation | 374 | 83 | 25 | 82 | STOP_harness |

Indices are zero-based. The continuations emitted a tool call, an assistant end
marker, a fabricated user/tool-result turn, and a later assistant JSON block.
The two parsers disagreed; the guard prevented a weight-selection claim. Every
run preserved all 426 original parameter hashes. See the separately generated
[postmortem](../experiments/direct-tool-finalization-cycle2-20261108/ORIGINAL-STOP-POSTMORTEM.json).
The original gate SHA-256 is
`824a514f9a8c9b2944387df1f88da48c6e14ab412ac4b0afdeef72d49489a77f`.
Neither a duplicate-call prefix nor the later JSON is a completed valid trial;
no stream is trimmed or relabelled to obtain a pass.

This limitation belongs to the shared observer's older generation profile.
Historical records retain their original producer and scope; they are not
upgraded by the newer verifier. The closed interval-study negative verdict is
unchanged and supplies no coordinates or activation donors for this new cycle.

## Repair and verification

- Compute the model/tokenizer EOS union once through the same helper used by
  completion reporting and pass it explicitly to `generate`.
- Require the first generated EOS token to be the terminal output token. The
  host independently recomputes this from raw token IDs. Preserve actual first
  stop index and terminal status in the hashed generation artifact.
- Reject internal chat-control markers before semantic tool parsing. Backend
  output past a requested stop is an integrity failure, not a neural repair.
- Keep exact-cap EOS complete, and keep non-EOS caps and unexplained short stops
  censored. Preserve all interruption and all-parameter postchecks.

Five new CPU regressions cover requested versus reported EOS, continuation past
the first stop, host evidence, fabricated turns and exact-cap completion. Four
failed before the fix; the exact-cap control already passed. A separate real
Transformers tiny-model test forces a chat EOS different from the checkpoint EOS:
the default continues for three tokens, while the explicit union stops after
one. This is a synthetic API test, not model-quality evidence or weight fitting.

The corrected `.54` suite passed **416 tests: 399 passed, 17 skipped**, zero
failures/errors, in 52.761 seconds. **Ten separate CPU runtime tests passed** in
7.600 seconds, including that actual generation-stop check. The seven remaining
unrun tests concern legacy training/PEFT/export APIs. No training environment was
started or modified. The tokenizer-only preflight authenticated all twelve files,
confirmed both EOS IDs, and retained the same 317/374-token inputs with call-ID
invariance and no language-model generation.

## Prospective amendment and cost custody

The [Sol xhigh review](../plans/direct-tool-finalization-cycle2-stop-repair-review-20261108.md)
is separate from the original proposal. The owner adopted one prospective
technical correction after observing the broken old stream and before corrected
scores. This is explicitly **not** an all-data-blind original preregistration.
The scientific hypothesis, exact two-episode gate, semantic cues, feature/support
rules, strengths and control thresholds remain unchanged.

The amended discovery cap is thirteen original-4B generation invocations:
original four plus one corrected tranche of four plus the unchanged five-case
semantic tranche. The 21,600 charged worker-second and 22-diagnostic caps remain.
All original and failed work stays charged: the
[prior ledger](../experiments/direct-tool-finalization-cycle2-20261108/BUDGET-LEDGER-PRIOR.json)
contains 1,751.884156 seconds; the corrected source's
[qualification ledger](../experiments/direct-tool-finalization-cycle2-20261108/QUALIFICATION-LEDGER.json)
adds 91.829473 seconds before any corrected generation. These costs include
qualification, authentication/loading, generation, postchecks and exit. The CPU
suite walls are conservatively charged even when overlapping another own test.
Tiny random-model tests have no official 4B discovery generations, but their
wall cost is charged.

`direct_tool_cycle54.py` captures both ledgers before execution, checks the whole
corrected tranche fits, and retains a durable one-use same-user claim. No reset,
free retry, old capsule reuse or additional corrected replay is admitted. Another
corrected harness fault ends the cycle. Fresh source, protocol, preflight and
initial-generation capsules are mandatory. The corrected protocol is
`configs/direct-tool-finalization-cycle2-chatstop.protocol.json`, SHA-256
`d44e668c125c3e3fb28ea756dcde83f0facbc7ec1d8a5e9fe589944152edec15`.

At this repair freeze, corrected BF16 reproduction, later causal controls,
weight derivation and independent paired quality are **`NOT_RUN`**. The independent
Sol max custodian has a public contract and has created no hidden cases. No
weights were edited, no LoRA/training occurred, no other jobs were controlled,
and no cleanup or provider-default change was performed.

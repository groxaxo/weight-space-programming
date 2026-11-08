# Completed practice sequence

The six-call screen completed, selected group 2 by its frozen rule, and left all
426 parameter tensors unchanged. The eight-call second stage completed and
passed nine checks, but failed the reverse joint-margin criterion. Its verdict
is `STOP_controls_failed`. No weight candidate or independent model-quality
result follows from these data.

- [Stage 1 raw result](causal-stage1.result.json)
- [Frozen Stage 2 request](CAUSAL-STAGE2-REQUEST.json)
- [Stage 2 raw result](causal-stage2.result.json)
- [Separate control arithmetic audit](CONTROL-ARITHMETIC-REVIEW.json)
- [Selected surgeon's interpretation](SURGEON-CONTROL-REVIEW.md)
- [Overall machine-readable findings](../direct-weight-20261108.json)
- [Overall report](../direct-weight-20261108.md)

Reverse transfer's next greedy operator was `>=`; its correct-minus-incorrect
joint continuation margin was +0.8176. The frozen criterion required a negative
joint margin as well, so the greedy result alone does not pass it. This preserves
positive evidence of conditional influence together with the failed gate.

The [separate arithmetic checker](recompute-controls.py.txt) uses recorded joint
probabilities, parameter hashes, request/protocol identity and actual norm/support
measurements. It does not import the producer's gate or run Qwen. To rerun it,
copy `causal-stage1.result.json`, `causal-stage2.result.json`,
`first-semantic-probe-02.result.json`, both `CAUSAL-STAGE*-REQUEST.json` files and
`CAUSAL-PROTOCOL.json` into a **new** directory; run the checker with that
absolute directory as its one argument. It publishes a new audit there without
overwriting an existing one. This is integrity/control verification, not an
independent quality evaluation.

The [Linux source transcript](linux-cpu-regression-transcript.txt) records 360
checks, OK with 14 skips. The seven runtime checks among those skips passed
separately: [five dtype/loader/reference checks](original-dtype-runtime-transcript.txt),
[the tiny probe control](first-semantic-runtime-transcript.txt) and
[the tiny additive control](causal-preflight-tests.json). Seven historical
training-library checks remain intentionally unrun. The actual
[causal tokenizer/runtime proof](causal-preflight-proof.json) precedes the full
model experiment.

`FILES.sha256` preserves the initial copied record index. `MANIFEST.sha256`
binds the final public record set, excluding itself. Supplementary reviews were
added without rewriting any raw run or the finding-record hash snapshot.

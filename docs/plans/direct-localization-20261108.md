# Sol xhigh: conditional localization before a 4B weight edit

Surgeon: `gpt-6.1-sol`, reasoning `xhigh`, explicitly selected by the user on
8 November 2026. This is a scientific review of the initial practice pilot,
not independent model-quality evaluation. No holdout was supplied to the surgeon.

The initial two identical greedy replays can establish runtime fidelity and
repeatability. They are not independent accuracy observations. The captured
last-prompt-token MLP activations predict the first generated token; they do not
identify a later code decision. They cannot justify selecting weight coordinates.

The exposed historical `half_open_overlap` failure rejected valid empty intervals
with `start >= end`. Its answer was 290 tokens, so a 192-token CPU pilot would risk
budget censoring. A new 384-token diagnostic protocol is frozen separately before
generation. The exact CPU BF16 runtime must reproduce its own completed failure;
historical Q8 client behavior cannot be assumed to carry over.

The next experiment is conditional on that reproduction:

1. Grade the completed generated code using the exposed assertions in a separate
   execution sandbox. Preserve outputs, token limits and the exact failure class.
   A pass or censored answer does not establish the proposed semantic defect.
2. If the strict/non-strict guard error occurs, freeze a decision-level practice
   protocol. Locate the exact prefix immediately before the incorrect operator.
   Inspect tokenization: `>` versus `>=` may require complete sequence scoring.
   Preserve the full prompt and assistant prefix; do not silently truncate it.
3. Compare that fixed prefix under the original prompt and a semantic reminder
   that equal endpoints are valid and empty. Record conditional log probabilities
   for both operators and bounded MLP input/output observations at the actual
   decision position, plus declared residual observations. A prompt-cue effect
   alone is not a neural repair.
4. With parameters unchanged, test same-model activation replacement from cue
   context into original context. Reset all cache/state between arms and hash
   parameters afterward. Start with four predeclared contiguous eight-layer
   groups covering all 32 layers, then subdivide only by a frozen rule. Record
   interactions if subdivisions lose the effect.
5. Require an unhooked baseline, self-to-self no-op, strengths `0`, `0.5`, `1`,
   two fixed-seed support/norm-matched random directions, an irrelevant nearby
   position and reverse transfer. Predeclare the correct/incorrect operator
   log-probability margin and next greedy choice as diagnostic outcomes.
6. Replicate on a declared practice family: both interval operands, paraphrased
   forms and a contrasting specification where zero-width intervals are invalid.
   These remain exposed cases. Do not fit weights to desired logits or inspect
   final evaluation material.

First time one fixed-prefix replay under measured CPU admission. At most two
contexts per job are initially justified; additional controlled replays must fit
the 10 GiB, one-core, current 1,800-second profile and host reserve. Sparse observations at
one position need approximately 0.72 MiB for both MLP vectors across all layers.
Unavailable capacity is reported; other jobs are never preempted.

Even a positive activation-transfer result shows that a component carries
causally relevant information in these contexts. It does not establish defective
weights or derive an allowed sparse BF16 intervention. A weight candidate needs
a reproducible 4B mechanism, evidence linking its read/write geometry to the
failure and a reviewed algebraic derivation without fitting. If those conditions
fail, preserve the official parent and report no justified repair. No LoRA or
training fallback is allowed.

Only after a candidate exists: run paired BF16 development controls, freeze the
patch/code/protocol, then use a separate custodian for untouched paired final
evaluation. Previously exposed HE9/client fixtures remain regression evidence.
No result from this initial pilot qualifies or promotes a model.

## Subsequent fixed protocol

The first failure reproduced twice. Both generated solutions failed the exposed
empty-interval assertions. A length/format control and an earlier semantic
assertion probe subsequently passed their preregistered diagnostic thresholds,
with every loaded parameter unchanged. These conditional prompt results do not
identify bad weights.

The separately frozen [finite causal protocol](../experiments/direct-weight-20261108/CAUSAL-PROTOCOL.json)
now supersedes the optional replacement sketch above for this hypothesis. It uses
semantic-minus-equal-length-control additive directions, exactly six then eight
forwards, a fixed four-group selection rule, actual BF16 norm/support checks,
fixed random signs, a verified code-fence token as the position control and
reverse transfer. It permits no adaptive subdivision or reselection after scores.
No-group, failed, unmatched or incomplete controls stop weight work. A pass still
requires a separate non-fitting geometry derivation and practice-family checks
before any sparse candidate. The [study report](../experiments/direct-weight-20261108.md)
records actual outcomes separately from this plan.

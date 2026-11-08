# Frozen direct-weight practice records

These are exposed diagnostic records from the 8 November 2026 study. None is an
independent model-quality evaluation. The unchanged official parent generated
the patient; no weight candidate is contained here. The study uses the
user-selected `gpt-6.1-sol` surgeon with `xhigh` reasoning.

- `PRACTICE-*`: original two-replay request/protocol and first completed 294-token
  patient, including actual input/output IDs. Both replays produced the same
  incorrect solution and failed separate exposed assertions.
- `DECISION-PROBE-*` and `semantic-probe-01.result.json`: original-prompt versus
  semantic-cue contrast at the later executable guard. Its prefix already
  contains an incorrect docstring definition.
- `NEUTRAL-*`, `neutral-probe-01.result.json` and
  `neutral-diagnostic-gate.json`: the separately frozen length/format control and
  repeated baseline. Control wording was selected by tokenizer counts before
  observing its model score.
- `FIRST-SEMANTIC-*` and `first-semantic-probe-02.result.json`: earliest incorrect
  semantic assertion, with exact patient replay and three contexts. All 426
  before/after parameter hashes are retained in the result.
- `CAUSAL-*`: fixed activation-only stages and Stage 1 request. Passing the
  contrast gate does not imply passing the activation controls.
- `SURGEON-PROVENANCE.json`: initial invocation snapshot. Later parent integration
  changed source/docs; this is not the final source inventory or a quality claim.

`FILES.sha256` binds the initial copied record set. A later manifest may include
additional results while preserving this initial index. Record files are never
rewritten to attach newer source identities or improved outcomes.

Protocols preserve historical limits, source hashes and failure chronology.
The current broker caps each worker at 1,800 seconds and probe/causal budgets at
1,700 seconds. Earlier completed jobs retained their larger declared maxima,
but their measured runtimes stayed below 1,800 seconds. Reproduction with
changed sources/runtime requires a new protocol identity, never relabelling a
historical run.

The repository contains numeric hashes and scores, not the model checkpoint or
activation archives. Full original inputs, numeric observations, frozen worker
sources, partial outputs and failed attempts remain preserved on `.54` under
the isolated study worktree. The [report](../direct-weight-20261108.md) separates
observed evidence, unrun acceptance stages and limitations. The
[method guide](../../direct-weight-editing.md) documents the worker and patch
interfaces. Use no untouched holdout material in these practice interfaces.

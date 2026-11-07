# Formatted semantic repair follow-up — 8 November 2026

This recipe is prepared after round one's development result and before any
testing of this new candidate. Round one is retained as a rejected experiment:
10/14 development passes versus base 12/14, despite held-out loss improving from
0.262717 to 0.116451. Its `bounded_chunks` learned the boolean guard but emitted
an invalid compound statement after a semicolon; escaping and version comparison
also regressed. The first pair's actual client trials must finish and be audited
before the live source is changed or this recipe is launched.

## Change and hypothesis

Return to the stronger released MLP, not the rejected round-one checkpoint.
Reduce learning rate from 3e-5 to 8e-6. Keep rank eight, alpha sixteen, MLP scope,
one full epoch, accumulation eight, 1,536 complete tokens, NF4 training, fresh
optimizer and seed 20261108. Exact parent factor equality is mandatory. Select
the lowest held-out-loss saved checkpoint before native testing.

Reuse the ten separate exercise families, expanding valid Python to ordinary
indented blocks with `ast.unparse`. Verify identical ASTs before and after;
invalid source is rejected, not repaired silently. Apply this formatting to
the original authored code replay too, retaining its original hash and task
IDs and preserving all public source pins. The altered readability instruction
changes presentation, not the requested program's semantics.

Add actual offline broken-check → edit → passing-check feedback to repair
trajectories. Add report-only observations of both broken and fixed states:
the model must copy the measured exit/failure count and stop. Common generic
check tool names and varied result field names teach the request's semantics.
The authored assertions are split into independent methods with fresh setup,
so broken controls produce varying measured failure/error counts. Adjacent call
and mutation postconditions stay in the same method; a failing regression
exposed and prevented a vacuous split before any model training. Targets and
outputs are the separate toy exercises, not the acceptance task or
its canonical answer. File/edit feedback remains simulated; unittest feedback
comes from the pinned offline Docker controls. All code variants, interfaces,
repair states and reporting states of a family share one task ID and split.

There are 600 new records (360 readable code/repair records and 240 status
observations), plus 640 pinned public and 288 original authored replay records
whose code formatting is normalized. The corpus manifest fixes final split
counts and complete sequence lengths before launch. No truncation is allowed.

## Unchanged gate and operations

Fresh paired base versus new candidate: full 262,144 context, 28,192 output
default/cap, Q8 K/V, non-thinking instruct sampler and seed 20261106; all fourteen
development cases and both OpenCode/Hermes file/project trials. Keep the same
tests, approvals, client versions, source freeze, turn limits and deadlines.
Require a strict development win, no lost base pass, `bounded_chunks` pass,
Hermes project pass, all four clients passed and complete executable/isolation/
identity/attribution/native-budget evidence. No model replacement from loss
improvement alone, and no benchmark-specific answers in the training data.

The tasks remain development-informed, not unseen evaluation. Do not relabel
the historical 8,192-token selected HE9 results as tests of this checkpoint.
Only a measured improvement may justify a separately documented experimental
model publication; root weights remain protected until the replacement gate
passes. Use existing `.54` GPU ownership/restoration controls and preserve
Bonsai, Moss, canonical memory, protected runtimes and historical VM evidence.

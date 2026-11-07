# Qwen semantic repair round — 8 November 2026

This protocol precedes training and model evaluation. The requested target is
the existing Hermes project failure and the released MLP's `bounded_chunks`
regression, under the corrected 28,192-token profile. Compact development and
client tasks are tuning-informed; they are not an unseen capability benchmark.

## Fixed recipe and selection

Start from the authenticated released MLP, not the weaker continuation. Use the
official pinned Qwen3.5-4B base, NF4 double quantization, rank-8/alpha-16 MLP-only
FP32 LoRA factors, 1,536-token complete conversations, batch one, accumulation
eight, one full epoch, seed 20261108 and learning rate 3e-5. Select the saved
checkpoint with lowest held-out loss before any native model testing. A lower
loss does not qualify an adapter. The optimizer starts fresh; exact parent
factor equality must be recorded before training.

Replay the original 640 public and 288 authored records and add separate
semantic repair exercises. The new generator teaches strict integer validation,
one-shot iterator materialization, duplicate detection, original-order priority,
and grounded read/edit/test/stop sequences. It does not import the acceptance
solutions, test files, historical HE9 prompts, reference answers or hidden tests.
All variants and code/tool interfaces of a semantic family share one task ID
and split. Execution controls run broken and repaired implementations in the
digest-pinned offline Python Docker image. File-edit responses are simulations;
unittest outcomes are actual offline observations. The generated manifest fixes
counts, file hashes and maximum length before training; truncation is forbidden.

## Paired gates

Run fresh base and candidate arms sequentially on `.54`, with identical runtime,
262,144 context, 28,192 output default/cap, Q8 K/V, non-thinking instruct sampling,
seed, isolated file/project fixtures, client versions, approvals and deadlines.
Run all fourteen development cases and both OpenCode/Hermes file and project
trials in each arm. Audit native requests and complete request attribution.

A replacement must pass `bounded_chunks`, preserve every base-passed development
case, strictly exceed the base development score, pass the Hermes project and
all four client trials, and have valid containment, source/weight identity,
independent test and request-attribution evidence. Acceptance tests, client turn
limits and timeouts are unchanged. Training loss alone cannot pass the gate.
No full HumanEval run is substituted for the selected Experimentos workload.
Any fresh historical HE9 replay remains separately identified by its original
8,192-token protocol and must be reported separately from this 28,192 profile.

## Publication and operations

Preserve the released weights unless a replacement passes the measured gate.
Record all failed candidates and any follow-up recipes before testing them.
Publish the report, reproducible code and appropriate Hugging Face updates,
then push `master` as explicitly requested. Do not describe a failed gate as a
successful assistant promotion. Pause only the known Bonsai service through the
existing GPU lease, restore it after every phase including failures, and preserve
Moss, protected runtimes, canonical MemPalace and the historical VM disk.

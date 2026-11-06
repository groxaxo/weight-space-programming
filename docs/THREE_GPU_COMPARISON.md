# Three-way RTX 3090 comparison

## State on 7 November 2026

Three real GPUQ jobs were submitted. They each request one exclusive physical GPU, with flexible placement by the existing scheduler. They may run concurrently when capacity permits; they do not share a card or require all three to become available simultaneously. No existing GPU workload was stopped, unloaded, or rerouted. At submission the cards were occupied and the available-RAM gate was also closed.

| Arm | GPUQ job ID | Requested GPU count |
|---|---|---:|
| Original Qwen3.5-0.8B, thinking enabled | `ab865a58c626` | 1 |
| Published repaired Qwen3.5-0.8B, thinking enabled | `ff5e96939107` | 1 |
| Original Qwen3.5-0.8B, thinking disabled | `318fe60bcc3d` | 1 |

**Latest check, 7 November 2026:** all three jobs were admitted, loaded their model on separate RTX 3090s, and entered generation. They were then cancelled via SIGINT at approximately 08:49 NZDT, before the first batch was saved. No accuracy or speed result exists. The cancellation initiator is not identified. See [interruption evidence](usage/benchmark-interruption-20261107.md) and [actual token accounting](TOKEN_USAGE.md). The submission-time resource observations above are historical, not the current queue state.

## Frozen protocol

The dataset is the full 500-question test split of `HuggingFaceH4/MATH-500`, pinned to revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`. Its immutable Git/LFS object identity and downloaded SHA-256 were checked. All three arms use four samples per question: 2,000 samples per arm and 6,000 overall.

Weights remain BF16. Every arm uses temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5, repetition penalty 1, and a 32,768 generated-token maximum. There is no early repetition stop. The engine context is 34,816 tokens. The longest rendered prompts were measured at 857 tokens with thinking and 859 without; no question or output budget is truncated to fit. Seeds derive from the shared question ID/sample index. The non-thinking arm uses matched sampling, not separately optimized non-thinking defaults.

The existing vLLM 0.24.0 / Torch 2.11.0 / Transformers 5.12.1 runtime is reused through a project overlay virtual environment. Only the small pinned math-grading dependencies were installed into the overlay. The shared runtime was not upgraded. Offline vLLM workers are used, not new HTTP services or proxy routes. Each worker checks its queue assignment before any CUDA-capable imports, honors the assigned UUID, and requires exactly one RTX 3090.

The evaluator verifies the original model and, for the repaired arm, the reconstructed checkpoint before loading. It checks code, dataset, dependency versions, per-sample seeds, and manifest identity. A run is bounded to 24 hours per attempt by the queue command. Completed samples are atomically saved and can be reused after an explicitly inspected retry. Failed or incomplete runs are never reported as full benchmark success.

## Resource admission

`scripts/gpu_admission.py` implements the actual GPUQ `resource`, `check`, and `acquire` contract. Jobs wait below 12 GiB free disk. An 8 GiB RAM reservation is recorded per admitted evaluation in addition to an 8 GiB available-memory floor. This is intentionally conservative because currently resident memory also remains counted in `MemAvailable`. The running evaluator checks for at least 8 GiB free disk and available RAM between batches, and stops only its own attempt on failure.

Waiting jobs do not launch polling GPU processes. The existing queue owns GPU availability checks, reservation, admission, and cancellation. The host's pre-existing idle watchdog remains unchanged. No GPU power limits or machine queue settings were changed.

## Results and recovery

Working checkout:

```text
/home/op/work/weight-space-programming-gpu-eval-20261107
```

Frozen run:

```text
runs/math500-three-arm-20261107/
  manifest.json
  queue-receipt.json
  original-thinking/{status.json,records/,summary.json}
  repaired-thinking/{status.json,records/,summary.json}
  original-nonthinking/{status.json,records/,summary.json}
  comparison.json
  comparison.md
```

The combined report is created after all three complete. It reports sample accuracy, question-level bootstrap intervals, paired accuracy differences, length truncation, extraction/grading failures, generated-token totals/distribution, runtime metadata, and actual GPU identities. Generation and wall timings are recorded per attempt; resumed runs must not use last-attempt time as total-run throughput. These independently queued runs are not a controlled GPU speed experiment.

```bash
gpuq show ab865a58c626
gpuq show ff5e96939107
gpuq show 318fe60bcc3d

cd /home/op/work/weight-space-programming-gpu-eval-20261107
.eval-venv/bin/python scripts/compare_gpu.py summarize \
  --manifest runs/math500-three-arm-20261107/manifest.json
```

The summarize command is CPU-only and returns a waiting status until complete. Read a failed job's logs before retrying it. Do not submit duplicates, overwrite this manifest, edit frozen runtime source while its jobs are pending, or remove the registered worktree. Use a new run ID/manifest for a changed experiment. Checkpoint binaries, dataset content, raw generations, queue state, and virtual environments remain outside Git.

## Validation performed before submission

- 62 CPU tests passed: the previous 40 plus 22 evaluation/admission/tokenizer regression tests.
- All 500 gold answers parsed with the pinned math checker; a symbolic equivalence self-test passed.
- Both thinking templates were rendered and all prompt lengths checked.
- vLLM engine/sampling argument construction and architecture registration passed without allocating a GPU.
- Original and repaired checkpoint verification passed again.
- Preflight caught a Transformers tokenized return-type difference; the runner now renders a string explicitly and encodes it without adding duplicate special tokens. Three regression tests cover this before queue submission.

Evidence is stored in `docs/reviews/evaluation-*-20261107.*`. This is engineering validation, not benchmark accuracy.

## Sol 6.1 Max plan

`docs/plans/sol-6.1-max-20261107.md` is the completed output of an actual `gpt-6.1-sol` / `max` Codex invocation. Model/effort, prompt digest, completion usage, and zero tool events are recorded in the accompanying provenance JSON. The run was plan-only with source supplied inline; the planner did not receive evaluation questions or answers and did not modify or execute code.

The plan was written while this targeted evaluation scaffold was being prepared. Its statement that evaluation was not implemented describes the prior merged baseline. This branch now supplies a narrowly scoped queued comparison, not the proposed general research framework. Autonomous orchestration, frontier budgets, activation extraction, general serving, and a sandbox for agent-written code remain planned, not implemented.

The plan recommends strict schemas and immutable edits; a resumable ledger and bounded provider broker; explicit Astra escalation; isolated CPU execution before enabling agent code; queue-owned GPU workers; practice-only extraction/iteration; and sealed final evaluation. KVM availability, stronger holdout custody, provider accounting guarantees, and one-GPU runtime capacity remain acceptance gates, not verified assumptions.

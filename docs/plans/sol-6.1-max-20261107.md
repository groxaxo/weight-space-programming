# Weight-Space Programming implementation plan

All observations below come from the supplied context. Nothing has been executed or independently rechecked. Filenames, interfaces, limits, and milestones below are proposals.

## 1. Evidence and current status

**Observed:**

- Private repository: `groxaxo/weight-space-programming`; PR #1 merged at `17611902e37ff76e7bf0d5e7b3012e8bd55fffbe`.
- Deterministic CPU reproduction exists; 40 tests passed twice. Published p50c edited tensor hashes match, with 130,188 BF16 entries changed.
- Verified checkpoints exist under `/home/op/experiments/weight-space-programming/models/{base,rebuilt}`.
- Autonomous orchestration, budgets, extraction, evaluation, serving, and isolated agent-code execution are absent.
- Host: i7-12700KF, 128 GB RAM, three 24 GB RTX 3090s, approximately 31 GB disk free. All GPUs are occupied.
- Available runtime: `/home/op/miniforge3/envs/qwen38-orca`, vLLM 0.24.0, Transformers 5.12.1, Torch 2.11.0.
- Three separate offline evaluation jobs are planned/in progress, without verified results: original-thinking, repaired-thinking, original-nonthinking.

**Unverified assumptions:** usable KVM isolation, exact provider model identifiers and usage semantics, one-GPU capacity at the required context length, dataset exposure status, and current resource availability. Validate these before dependent work.

## 2. Architecture and trust boundaries

Use a trusted controller with a bounded research loop:

`propose → validate → sandbox → validate edit → practice evaluation → select → freeze → sealed evaluation → finalize`

GPT-6.1 Sol with MAX reasoning is the proposed default. Configure exact supported provider identifiers; refuse silent substitutions. Astra requires an explicit operator grant specifying scope, rationale, expiry, and budget. Failure or exhausted Sol retries does not authorize escalation.

Separate capabilities:

- **Provider broker:** holds frontier credentials; authorizes requests and budgets.
- **Controller/ledger:** owns state transitions, policies, and provenance.
- **CPU sandbox:** executes untrusted agent-written code.
- **Artifact broker:** accepts bounded data artifacts.
- **Trusted GPU workers:** run reviewed extraction/evaluation implementations through GPUQ.
- **Sealed evaluator:** independently holds final questions and labels.

Agents cannot directly execute host commands, submit GPU jobs, change grading, access credentials, or inspect heldout data. Prompts, generated code, dataset text, and returned artifacts remain untrusted.

## 3. Proposed interfaces and schemas

Use versioned, strict JSON schemas rejecting duplicate keys, unknown fields, invalid dimensions, and oversized payloads.

| Proposed files | Contract |
|---|---|
| `schemas/experiment.schema.json`, `configs/research.example.json` | `ExperimentSpec`: phase, checkpoint/data/protocol/runtime hashes, seeds, limits, budgets, selection rule |
| `src/wsp/controller.py`, `providers.py` | Structured proposals; exact model/effort mapping; explicit escalation grants |
| `src/wsp/budget.py`, `schemas/budget.schema.json` | Reservations, usage settlement, unknown usage, pricing snapshot, hierarchical limits |
| `src/wsp/ledger.py`, `state/research.sqlite3` | Runs, attempts, events, reservations, external job IDs, artifacts, finalization records |
| `src/wsp/edits.py`, `schemas/edit.schema.json` | Parent digest, tensor names/shapes, sorted indices, BF16 replacement words, edit limits |
| `src/wsp/extract.py`, `schemas/activation.schema.json` | Exact hooks, token selection/masks, dtypes, seeds, dataset lineage, shard limits |
| `src/wsp/evaluate.py`, `grade.py`, `schemas/evaluation.schema.json` | Frozen generation/grading protocol; per-sample outputs and outcomes |
| `src/wsp/gpuq.py`, `worker.py` | Trusted job manifests; GPUQ submission/reconciliation; owned worker lifecycle |
| `src/wsp/sandbox.py`, `broker.py` | MicroVM resource policy and allowlisted artifact exchange |

Every artifact records its content hash, producer revision, input hashes, schema version, and owning run. Keep runtime state, credentials, checkpoints, and generated datasets outside Git.

## 4. Persistent ledger and idempotence

Use SQLite transactions, WAL, foreign keys, and durable event records. Derive logical run/sample identities from canonical immutable specifications; distinguish logical work from execution attempts.

States include validated, budget-reserved, queued, running, completed, failed, usage-unknown, and finalized. Persist GPUQ IDs and report queued/running/completed separately.

Use a transactional submission outbox. After an ambiguous submission, reconcile GPUQ before resubmitting. If GPUQ lacks submission idempotency, fail closed on unresolved ambiguity rather than promising exactly-once execution.

Workers claim logical work using fencing tokens. Publish results atomically without replacement; duplicate attempts cannot replace accepted outputs. On restart:

- Reconcile external jobs, calls, reservations, and artifact hashes.
- Resume missing samples without regenerating completed ones.
- Preserve uncertain charges and live ownership.
- Remove staging artifacts only after proving ownership and inactivity.

## 5. Token and spending budgets

Maintain independent run, experiment, model, request, and attempt budgets. Track frontier tokens separately from local-model generation tokens, GPU time, wall time, and artifact bytes.

**API mode:**

- Reserve worst-case input, maximum billed output, and bounded additional charges before dispatch.
- Assume uncached input pricing when reserving; recognize cache discounts only from authoritative usage.
- Record input, cached input, visible output, reasoning, and raw provider usage. Provider adapters define overlapping fields so reasoning is never counted twice.
- Use timestamped, operator-approved pricing snapshots and integer monetary units. Unknown pricing blocks monetary-budget mode.
- Disable hidden SDK retries. Each retry, fallback request, and concurrent attempt requires a separate atomic reservation.
- Retain conservative reservations after cancellation, interrupted streaming, timeouts, or missing usage; reconcile later.
- Disable unpriced tools and endpoints without reliable usage bounds.

The ledger must guarantee that authorized reservations stay within configured limits. Account-wide billing guarantees additionally require suitable provider controls; local accounting cannot constrain unrelated account activity.

**Subscription Codex mode:** invocation counts, deadlines, and exposed token limits can constrain activity, but logs may be incomplete or post hoc. Never infer or promise a hard dollar cap from them. Require explicit selection of this weaker accounting mode; never silently substitute it for API mode.

Start with one frontier request in flight. Raising concurrency must preserve transactional reservations across all shared budgets.

## 6. Immutable edits and dataset isolation

Preserve the existing reproduction pipeline’s narrow guarantees. Implement research edits separately.

Keep original and published-rebuilt baselines immutable. Each candidate names its immutable parent and records complete ancestry. Revalidate composed edits against the designated baseline.

Initially allow only the two locked MLP `down_proj.weight` tensors. Require unique sorted indices, exact shapes, BF16 replacement bits, finite values, and operator-defined sparsity/magnitude/norm limits. Record selected positions separately from actual changed positions. Reject undeclared edits, executable payloads, pickle, tokenizer/config changes, and changes outside allowed spans. Publish verified candidates without overwriting existing paths; prohibit writable hardlinks to baselines.

Register train/practice and sealed-heldout datasets with immutable hashes, provenance, access principals, and overlap checks. Holdout questions, labels, activations, and intermediate scores remain unavailable to agents.

MATH-500’s exposure status must be explicit. Existing evaluations cannot retroactively establish that it was sealed or unseen. If development exposes its results, classify it accordingly and select an independent final set.

Before final evaluation, freeze candidate, baseline, protocol, grading code, seeds, selection criterion, budgets, and reporting rules. Release results only after finalization. Subsequent tuning requires a new research cycle and independent holdout.

## 7. Activation extraction methodology

Use a trusted Transformers extractor, separately fingerprinted from vLLM generation.

For layers 15 and 19, resolve the exact runtime modules corresponding to the locked tensor names:

- `register_forward_pre_hook`: capture `down_proj` positional input 0, the post-gating MLP features.
- `register_forward_hook`: capture the projection output before residual addition.

Assert invocation counts and expected `[batch, sequence, feature]` shapes; reject unresolved modules or unexpected outputs.

Use evaluation/inference mode, disabled gradients/dropout, `use_cache=False`, local-only loading, and no remote model code. Preserve BF16 weights and native BF16 activations; store lossless BF16 tensors. Compute summaries in FP32.

Extract teacher-forced sequences from authorized practice data only. Pin tokenizer/template hashes, token IDs, attention masks, prompt/completion boundaries, BOS/EOS treatment, padding exclusion, and selected token positions. Capture complete feature dimensions for the declared positions.

Record Python/NumPy/Torch/CUDA seeds, batching, device UUID, runtime/kernel settings, and determinism limitations. Stream bounded shards with sample/token offsets and hashes. Reject resource overflow rather than silently changing token selection or precision.

## 8. Controlled evaluation

Adopt the existing scaffold only after reviewing its artifacts, protocol, and queued-job identities.

Freeze the three requested arms with:

- 500 immutable questions × four independently identified samples.
- Maximum 32,768 generated tokens.
- Temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5.
- No early loop stopping; pinned normal EOS/length behavior.
- Identical grading and remaining generation settings.

Use per-question/sample seeds shared across compared checkpoints. Record the intentional thinking-template difference. Estimate repair effects from original-thinking versus repaired-thinking; the nonthinking arm answers a separate comparison.

Maximum generation is **65,536,000 tokens per arm**, **196,608,000 across three arms**. Validate context capacity for prompt plus generation; never truncate questions or reduce budgets to fit.

Persist raw outputs, token counts, termination reason, answer extraction, grade, errors, timing, and GPU UUID. Report completeness, accuracy, truncation, parse/grade failures, generated tokens, and elapsed time. Predeclare failure treatment; never silently exclude failed samples. Use question-level paired uncertainty estimates.

Interrupted runs resume missing fixed samples under predefined retry rules. Report incomplete runs explicitly. Timing is descriptive unless hardware and execution conditions are controlled. No positive quality result is required for engineering acceptance.

## 9. GPUQ lifecycle and resource guards

GPUQ remains the sole GPU scheduler and admission authority. During implementation, read its policy and inspect actual interfaces before integrating.

Submit one exclusive whole-GPU job per worker, tensor parallelism one. No CUDA initialization, model allocation, or serving before admission. Unavailable telemetry or occupied devices leave jobs queued. Persist job IDs and reasons.

Use the admitted GPU UUID and lease-provided environment. Workers load verified local BF16 checkpoints, perform readiness checks, execute bounded work, flush outputs, and terminate their own process trees before lease release. Serving is experiment-scoped, privately reachable, and never left holding an idle GPU indefinitely.

Never stop, unload, reroute, or alter existing workloads, power settings, or leases. Do not introduce GitHub Actions or another scheduler.

Before submission and again after admission, check available disk/RAM against projected peak use, including outputs, activations, staging, VM storage, and logs. Proposed starting reserve floors: 10 GiB disk and 32 GiB available host RAM after projected demand; validate operational suitability first.

Prefer sparse candidate storage and quota-authorized materializations. Block insufficient-capacity jobs; do not silently reduce precision, context, or benchmark settings.

## 10. CPU sandbox threat model

Treat agent code as malicious: filesystem/credential theft, network exfiltration, subprocesses, fork bombs, native extensions, parser attacks, and resource exhaustion.

Propose a CPU-only KVM microVM with a pinned minimal guest. Verify availability first; fail closed if unavailable.

Provide:

- No NIC, host credentials, host sockets, shared home directory, GPU devices, or general host RPC.
- Read-only virtual input disks containing only approved data and baseline material.
- Bounded disposable scratch/output storage.
- External CPU, RAM, process, I/O, output-byte, and wall-clock limits under an unprivileged supervisor.
- Guest-side limits as additional protection.

Use a narrow artifact channel accepting only declared tensor patches and bounded diagnostics. Validate names, lengths, shapes, hashes, and numeric constraints in an isolated parser. Never import returned code or mount guest-written filesystems directly on the host.

Codex read-only settings and AST/keyword filtering are not security boundaries. Hypervisor/kernel compromise remains a residual risk; isolation updates require separate review.

## 11. Pull-request-sized milestones and acceptance gates

| PR | Dependencies | Deliverable and measurable gate |
|---|---|---|
| 1 | Existing merge | Schemas, locks, split registry, candidate validator. Existing 40 tests pass; malformed edits, unauthorized tensors, mutation, and overwrite attempts fail, including optimized Python. |
| 2 | 1 | Ledger, provider broker, budgets, controller. Crash/replay, concurrent reservation, cache/reasoning accounting, retry, unknown-usage, and unauthorized Astra tests pass. |
| 3 | 1–2 | MicroVM and artifact broker. Negative tests demonstrate blocked network, credential/host/GPU access, traversal, oversized artifacts, fork bombs, and deadline overruns. |
| 4 | 1–2 | GPUQ lifecycle and evaluation runner. Fake occupied/unavailable telemetry produces zero launches; admitted practice smoke test records one GPU, BF16, provenance, and cleanup. |
| 5 | 3–4 | Extraction and bounded practice loop. Hooks match direct captures; masks/offsets reconstruct selected rows; shard bounds hold; extraction leaves model hashes unchanged. |
| 6 | 2–5 | Sealed finalization and reporting. Leakage/access tests fail closed; frozen manifests cannot change; interrupted evaluation resumes without replacing results; finalization is idempotent. |

GPU-dependent gates wait for legitimate admission. None authorize disturbing occupied GPUs.

## 12. Open decisions and first steps

Resolve exact model identifiers, API versus subscription mode, budgets, Astra grants, edit limits, holdout custody, KVM support, capacity, and preregistered improvement criteria. Principal risks are data leakage, uncertain usage, activation volume, nondeterministic kernels, and indefinite GPU waiting.

At later authorized execution:

1. Confirm the merged checkout and identify existing evaluation manifests/job IDs.
2. Record the unresolved decisions in a reviewed experiment specification.
3. Implement PR 1, then PR 2; keep generated code disabled until PR 3 passes.
4. Run practice-only pilots through GPUQ before authorizing full evaluation or autonomous iteration.

This plan establishes engineering and research-validity gates; it claims no new benchmark result.
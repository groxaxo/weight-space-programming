# Qwen3.5-4B coding-assistant workflow

This track adapts the official Qwen3.5-4B checkpoint for coding and structured tool use. It produces a separate LoRA adapter and preserves the base files. A completed training run is a candidate; deployment and broad coding reliability are separate questions assessed through the paired evaluations and real clients.

## Pinned inputs and attribution

| Input | Immutable revision | Terms |
|---|---|---|
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Apache-2.0 |
| NVIDIA [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d` | CC-BY-4.0; retain NVIDIA attribution |
| Nous Research [Hermes function calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | `dae3e1d28cfbcf4b915c04ea1e072030529b4bda` | Apache-2.0 |
| OpenAI [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) | `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` | MIT; evaluation only |

The initial corpus contains 512 public coding examples and 128 complete tool conversations, split deterministically into 576 training and 64 validation examples. OpenCodeInstruct examples are filtered using the publisher's all-passing test metadata; that is a selection signal, not our own correctness guarantee. Invalid calls, incomplete conversations, duplicate prompts, and oversized examples are rejected or recorded. The import scripts read only those two public corpora, do not read private chats or user repositories, reject recognizable credentials, and do not import benchmark reference solutions or tests as training targets. This does not rule out related material already present in the public corpora.

Keep the [input attribution and modification notice](coder-attribution.md) with exported adapters and data.

Every prepared split has a SHA-256 and task identity list. The loader rechecks these and rejects task/prompt overlap between training and validation. This prevents exact task/prompt leakage within this corpus; it does not establish semantic disjointness or rule out public benchmark content in the base model's pretraining.

## Prepare on Linux / Python 3.12

The reconstruction environment remains small (`requirements.txt`). Use a separate CUDA environment for this track:

```bash
python3 -m venv .venv-train
.venv-train/bin/python -m pip install -r requirements-coder.txt
.venv-train/bin/python -m unittest discover -s tests -v
.venv-train/bin/python scripts/download_coder.py --proof reports/model-inputs.json
.venv-train/bin/python scripts/import_coder_corpora.py \
  --output assets/coder-corpus-v1 --code-count 512 --tool-count 128
```

The full [verified .54 environment lock](../configs/qwen35-environment-54.lock.txt) also pins transitive dependencies for the recorded Linux/CUDA run. Use that file instead of `requirements-coder.txt` when reproducing that exact environment.

The downloader checks every required file against the immutable Hub revision's LFS SHA-256 or Git blob identity and writes `reports/model-inputs.json`. The verifier requires every weight shard referenced by the authenticated index and every loader sidecar present in the snapshot. Training and serving reauthenticate that snapshot and load it offline with `trust_remote_code=False`. Existing proof files, datasets, model reports, and training outputs are not overwritten; choose a new output path for another experiment.

`configs/qwen35-4b.json` defines the seed, context length, adapter rank, learning rate, accumulation, and minimum usable split sizes. The recipe uses the model's native chat template. User, system, and tool-result tokens are masked out of the loss. Delimiter and credential-pattern checks cover nested tool arguments and tool schema/description strings as well as message content. Whole examples that exceed the context limit are recorded and dropped instead of being truncated through a tool call or answer.

## GPU phases on .54

The working checkout is `/home/op/work/weight-space-programming-20261106`. This host has one RTX 3060, no GPUQ installation, and an existing Bonsai service. Direct admission is limited to this observed host and requires at least 10,500 MiB of free GPU memory with no active compute. It refuses to stop a service on its own.

The phase supervisor supports an explicit `--allow-pause-bonsai` authorization. It records whether Bonsai was active, stops it for the requested phase, and restores it in a `finally` block. Run the supervisor in a transient systemd unit with its `--restore-only` command as `ExecStopPost`, so forced process termination also triggers restoration. Unit names and output directories must be unique.

`service-before-phase.json` records the initial state independently of the transient pause marker. Restoration uses this record too, so losing the marker cannot silently leave a previously active Bonsai service stopped. An unknown service-state query fails before the phase starts. All checkouts share `$XDG_RUNTIME_DIR/wsp54-gpu` (normally `/run/user/1000/wsp54-gpu`). Its exclusive lock and owner journal bind cleanup to a single lease. Completed callbacks are idempotent; an older callback cannot start Bonsai inside a newer GPU window. An abandoned lease is recovered under the host lock before another phase is admitted. After stopping Bonsai, a bounded 60-second readiness wait lets CUDA teardown and NVIDIA’s sampled utilization settle; model admission still requires at least 10,500 MiB free and zero utilization, and unknown GPU processes are never stopped.

```bash
ROOT=/home/op/work/weight-space-programming-20261106
OUT="$ROOT/runs/train-example"
systemd-run --user --collect --unit=wsp54-train-example \
  --property=RuntimeMaxSec=10800 --property=TimeoutStopSec=180 \
  --property=KillMode=control-group --property="WorkingDirectory=$ROOT" \
  --property="ExecStopPost=/usr/bin/python3 $ROOT/ops/run54_gpu_phase.py --restore-only --output $OUT" \
  /usr/bin/python3 "$ROOT/ops/run54_gpu_phase.py" \
  --phase train --output "$OUT" --allow-pause-bonsai
```

Use `--phase smoke` for two optimizer steps, or `--phase evaluate` for the model tests. Evaluation accepts `--adapter /absolute/path/to/training/adapter`, `--clients`, and `--humaneval`. A smoke adapter is explicitly refused as a serving candidate.

Use `--phase benchmark` to run HumanEval alone against the same serving identity. Client failures are recorded in `acceptance.json`; they do not prevent gathering the remaining benchmark evidence.

The supervisor limits host thread use and enables expandable CUDA allocation segments for varying sequence lengths. Training probes the longest training example using the actual backward/optimizer path at zero learning rate before the epoch. It periodically validates and retains checkpoints, selects the lowest validation loss, checks finite metrics, and records adapter checksums, source hashes, package versions, device allocation, and peak CUDA memory. Before model loading, training writes an authenticated `source/` archive of its scripts, operations code and recipe configuration. These hashes are recorded in `run-manifest.json`. A failed job is incomplete; inspect `phase-failed.json` and `training.log` before retrying.

Lower validation loss does not guarantee better downstream behavior. To evaluate an earlier saved checkpoint from a completed run, export it without changing the original artifacts:

```bash
.venv-train/bin/python scripts/select_coder_checkpoint.py \
  --training runs/train-02/training --step 18 --output runs/checkpoint-18/training
```

The exporter checks all saved tensor names against the original language-module targets, verifies finite weights, preserves source provenance, and uses that step's actual validation loss. It authenticates the frozen source archive and writes a separate unpromoted candidate through staging and atomic no-replace publication. A failed copy leaves no final destination, so the export can be retried. The selection script is preserved separately in `selection-source/`; it does not overwrite the parent recipe. Development results may guide checkpoint choice; HumanEval and client acceptance still require fresh measurements for the selected adapter.

The existing GPUQ route remains the allocation path on .51. `--direct-idle-gpu` is a specific .54 exception and does not permit bypassing .51's queue.

## Evaluate the actual candidate

The local endpoint is `http://127.0.0.1:18084/v1`. Its schema-2 identity advertises the authenticated model proof, adapter weights and configuration, training-result identity, runtime, quantization mode, effective context/output limits and serving-protocol hash. A separate per-process instance ID binds requests to a running server without making baseline/candidate profile identities process-specific. Context and output limits are enforced. Calls are serialized and use greedy generation in the evaluations.

Standalone top-level Qwen3.5 XML function calls are parsed and validated against the provided JSON schemas, then returned as OpenAI `tool_calls`. Fenced examples and valid Python containing tool markup remain literal coding output. Both ordinary JSON and SSE Chat Completions clients are supported. SSE currently emits one completed delta; it does not provide token-by-token latency. This is a measured local experiment server, not a multi-user production inference engine.

Generation explicitly stops at both the native chat boundary (`<|im_end|>`, token 248046) and the checkpoint EOS (`<|endoftext|>`, token 248044). The pinned checkpoint's default EOS alone does not stop an assistant turn at its chat boundary. The serving identity records the effective stop IDs, and regression tests cover this distinction.

```bash
.venv-train/bin/python scripts/evaluate_coder.py \
  --base-url http://127.0.0.1:18084/v1 \
  --image python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54 \
  --output reports/candidate-development.json

.venv-train/bin/python scripts/evaluate_humaneval.py \
  --base-url http://127.0.0.1:18084/v1 \
  --image python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54 \
  --output reports/candidate-humaneval.json

.venv-train/bin/python scripts/compare_coder.py \
  --baseline reports/baseline-development.json \
  --candidate reports/candidate-development.json \
  --output reports/development-gate.json
```

The development suite has eight executable Python tasks and six multi-turn tool scenarios, including dependent calls, error recovery, typed arguments, and untrusted file text. The gate requires a complete matching pair, at least 80% coding success, all tool scenarios passing, no per-case regressions, and an actual improvement. Different base revisions, quantization, inference runtimes, serving protocols, or evaluator implementations cannot be silently compared. HumanEval adds all 164 official tasks, with the same output budget and sandbox for baseline and candidate. Its public status prevents a claim of uncontaminated generalization.

For the complete local acceptance gate, also pass `--baseline-humaneval`, `--candidate-humaneval`, `--clients` (the candidate's `acceptance.json`), and `--training` (its `training-result.json`) to `compare_coder.py`. This requires all 164 paired tasks, no aggregate HumanEval regression, both installed clients making edits that pass independent tests, and a finite improvement in held-out validation loss. A missing or mismatched report fails closed. Weight/configuration checksums and the training-result fingerprint are mandatory. Each successful client result must also carry a verified request attestation from the current proxy. Historical schema-1 reports are rejected by the current gate; use their frozen source to inspect historical decisions, and regenerate baseline/candidate evidence for a new qualification run. Do not add identity fields to old reports retroactively.

Generated Python executes only in the pinned Docker image with networking disabled, a read-only root filesystem, an unprivileged user, bounded memory/CPU/processes, and a read-only fixture mount. Sandbox self-tests run before scoring. This isolates generated code for these evaluations; it is not a general policy for unrestricted agent shell commands.

The checker requires a completion exit marker after all assertions, so an ordinary `exit(0)` cannot bypass the tests and produce a passing score. The Docker integration tests cover correct code, incorrect code, `SystemExit(0)`, and `os._exit(0)`. Run them with `WSP_TEST_DOCKER_IMAGE` set to the pinned image; the CPU CI workflow sets this automatically. This completion check is for grading, rather than a guarantee against deliberate grader manipulation.

If only sandbox scoring changes during a long HumanEval run, `rescore_humaneval.py` can regrade every saved output without regeneration. It requires the original frozen runner source, verifies its hash, verifies the generation client and code extractor are unchanged, authenticates the exact dataset/tests, and records the parent report hash and both runner versions. Generation changes require a new run.

## OpenCode and Hermes

The [OpenCode example](../examples/opencode.json) uses `@ai-sdk/openai-compatible`. The [Hermes example](../examples/hermes-config.yaml) defines a separate custom-provider profile. Neither requires changing a user's existing default provider. The installed clients are tested against isolated fixture directories and empty client state, using file tools only. Both client harnesses route actual model requests through `ops/client_proxy.py`. It validates the model profile and server instance on every JSON response or SSE chunk, requires real completion requests, rejects provider drift/fallback, and records only request hashes/counts. Unsupported SDK capability-discovery routes return 404 and never count as model requests; they do not invalidate an otherwise verified completion. The harness independently tests the edited source in Docker and verifies that the client did not alter the fixture tests.

```bash
.venv-train/bin/python ops/smoke_clients.py \
  --base-url http://127.0.0.1:18084/v1 \
  --image python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54 \
  --output reports/client-acceptance
```

An OpenAI-shaped response alone does not establish either client's acceptance. Preserve `acceptance.json`, the actual CLI logs, the modified fixture, and its independent test result. Broader repositories, unrestricted terminal tools, larger contexts, other languages, and a default-provider rollout need their own measured evidence.

## Use the endpoint on demand

After reviewing the candidate's acceptance report, start a bounded session on .54:

```bash
cd /home/op/work/weight-space-programming-20261106
python3 ops/local_coder54.py start \
  --adapter /absolute/path/to/the/evaluated/training/adapter --allow-pause-bonsai
python3 ops/local_coder54.py status
# When finished:
python3 ops/local_coder54.py stop
```

The launcher uses the existing supervised service path, serves loopback port 18084, and limits a session to two hours. Wait for `status` to show the endpoint identity before connecting a client. Stopping the session or reaching its time limit restores Bonsai. The example client profiles connect to this endpoint; existing client defaults remain independently configured.

For an unchanged-base comparison, use `start --base --allow-pause-bonsai` instead of `--adapter`. The endpoint identity makes the selected base/adapter state explicit.

The optional project acceptance harness exercises two Python modules, file tools,
terminal test execution and independent grading. Start the local endpoint first,
then run this on .54:

```bash
.venv-train/bin/python ops/project_clients.py \
  --image python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54 \
  --output reports/project-client-acceptance
```

The default runs both installed clients in fresh state. Each gets a broken
dependency scheduler, may edit only its two modules, and must actually invoke
the supplied unittest command. The harness protects the fixture tests, grades
the final modules independently and records actual tool events. It observes the
trial's uniquely labeled containers and requires disabled networking, a non-root
user, capability and process limits, bounded memory and task-scoped mounts.
OpenCode runs inside the container and reaches the host loopback model API via a
temporary Unix socket relay. Hermes uses its per-session Docker file/terminal
backend with an explicit fixture mount and fresh profile caches. The harness
writes `terminal.backend: docker` and all container settings into that task-only
profile and uses `--ignore-rules`. In the measured Hermes managed-worker version,
`--safe-mode` bypassed the profile and an environment-only override did not bind
the actual terminal backend. The original attempt was stopped and preserved as
infrastructure evidence. The corrected configuration was verified through the
real managed CLI with a synthetic provider before a fresh model trial. Both paths
remove only their own test containers. Each client has a 1,000-second wall limit.
These project results are reported separately from the frozen paired development
and HumanEval gate; a transport control is not a model-quality result.

## CPU compilation and experimental native inference

An evaluated LoRA can be compiled into a separate BF16 checkpoint without loading
the entire model into PyTorch or allocating GPU memory:

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python scripts/compile_coder.py \
  --config configs/qwen35-4b-lr5e-6.json \
  --model-proof reports/model-inputs.json \
  --training runs/checkpoint-lowlr-18/training \
  --output runs/compiled-lowlr18-bf16
```

Use a new output directory. The compiler authenticates the original checkpoint
and adapter, computes one FP32 matrix update at a time, rounds to BF16 using
round-to-nearest with ties to even, and verifies every unedited byte before
publishing. `compile.json` binds the output to its inputs and compiler. Successful
compilation establishes artifact integrity; its numerical rounding and a later
quantization create an inference profile that requires fresh paired evaluation.

The optional native backend uses the existing read-only llama.cpp installation
on .54. `scripts/serve_coder_gguf.py` sends exact HF input token IDs to its loopback
completion API and decodes the returned output IDs with the same tokenizer. It
uses the existing request validation, native tool-call parser and completed-delta
SSE handler. The supervisor `ops/run54_gguf_phase.py` authenticates the converted
artifacts and runtime, applies the same GPU admission and Bonsai restoration
policy, and supports development, benchmark and serving phases. Its `--help`
describes the required conversion manifest and profile selection. Create that manifest with the committed converter after CPU compilation:

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python scripts/convert_coder_gguf.py \
  --llama-source /home/op/src/llama.cpp-prismml-mtp/src \
  --model-proof reports/model-inputs.json \
  --compiled runs/compiled-lowlr18-bf16 --profile candidate \
  --output runs/native-compiled-example
```

The command authenticates the source snapshot and compiled files, produces the Q8 base and candidate, preserves converter hashes, and atomically publishes `conversion.json` plus `compile-record.json`. Pass `--conversion runs/native-compiled-example/conversion.json --profile candidate` to the native supervisor.

For a separate Q8 base plus unmerged FP32 LoRA, export with:

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python scripts/export_coder_lora_gguf.py \
  --llama-source /home/op/src/llama.cpp-prismml-mtp/src \
  --config configs/qwen35-4b-lr5e-6.json \
  --model-proof reports/model-inputs.json \
  --training runs/checkpoint-lowlr-18/training \
  --output runs/gguf-lora18-f32
```

Build a manifest for the unmerged route with:

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python scripts/convert_coder_gguf.py \
  --llama-source /home/op/src/llama.cpp-prismml-mtp/src \
  --model-proof reports/model-inputs.json \
  --lora-export runs/gguf-lora18-f32 --profile candidate-lora \
  --output runs/native-lora-example
```

Add `--reuse-base /absolute/path/to/prior/conversion.json` to authenticate and reuse an existing Q8 base. Otherwise the producer converts a fresh base. Use the resulting `conversion.json` with `--profile candidate-lora`; the manifest binds the export and its base without hand-authored fields.

The exporter adjusts hybrid-head permutations on the LoRA factors in memory:
row permutations affect B and column permutations affect A. It verifies finite
FP32 factors and unchanged protected converter files. Golden controls compare
the factorized update with the dense permutation. The frontend requires the
native completion response to confirm that the expected adapter is active.

Native profiles use the separately declared [acceptance contract](native-acceptance.md)
and `scripts/compare_gguf_coder.py`; results from different protocol hashes cannot
be paired. The measured BF16-merged and FP32-unmerged experiments failed the
development quality requirements. They remain unpromoted, and no native
HumanEval qualification is claimed.

## Measured selection on .54

The official base scored 6/8 authored Python tasks, 6/6 tool scenarios and
128/164 HumanEval tasks, and passed both real clients' initial file-edit fixtures.
The lower-rate LoRA at step 72 scored 7/8, 6/6 and 126/164 respectively; its earlier
step-18 checkpoint scored 7/8, 6/6 and 127/164. Both passed the initial client
fixtures but failed the complete aggregate non-regression gate. Higher-rate
checkpoints also showed development regressions. Lower validation loss therefore
did not justify promotion. The unchanged base remains the selected profile;
all fine-tuned weights are retained as unpromoted research artifacts.

These measurements concern a small text-only Python workload on one RTX 3060.
They establish neither broad repository reliability nor equivalence to a larger
coding assistant. The delivery reports include the separate two-module client
trial, all failed candidates, model/data revisions and file hashes.

The [comprehensive report](experiments/qwen35-4b-20261107.md) and
[machine-readable findings](experiments/qwen35-4b-20261107.json) include the exact
checkpoint decisions and later integration corrections. `--safe-mode` in the
earlier file-only interactive example does not establish terminal isolation;
use the explicit task-profile project harness for contained terminal trials.

## Bounded end-to-end verification on .54

`ops/verify_pipeline54.py` uses 16 training and four validation conversations from the authenticated public corpus for a complete two-step control epoch. It freezes the recipe, exports checkpoint 1, exercises the base and control adapter through the development evaluator and actual OpenCode/Hermes clients, compiles all 248 matrices on CPU, exports FP32 factors, produces the native manifest and exercises native inference/clients. It records actual outcomes without treating the tiny control adapter as a qualified coding model.

```bash
ROOT=/home/op/work/weight-space-programming-20261106
OUT="$ROOT/reports/pipeline-verification-example"
systemd-run --user --collect --unit=wsp54-pipeline-example \
  --property=RuntimeMaxSec=5400 --property=TimeoutStopSec=240 \
  --property=KillMode=control-group --property="WorkingDirectory=$ROOT" \
  --property="ExecStopPost=$ROOT/.venv-train/bin/python $ROOT/ops/verify_pipeline54.py --output $OUT --restore-only" \
  "$ROOT/.venv-train/bin/python" "$ROOT/ops/verify_pipeline54.py" \
  --output "$OUT" --allow-pause-bonsai \
  --base-conversion /absolute/path/to/existing/base/conversion.json
```

A pinned model/corpus, the full training environment, installed clients, Docker, and the protected native runtime must already exist. Use a unique unit and output directory. `--restore-only` covers both the HF and native lease outputs. `verification.json` distinguishes pipeline execution, client/code quality results, and whether HumanEval was regenerated. Add `--humaneval` only for a fresh all-164-task measurement of both HF profiles; it substantially extends runtime, so raise the transient unit’s `RuntimeMaxSec` to `21600` for that mode. The ordinary bounded control does not update the original HumanEval scores or promote its adapter. The [nine-finding resolution record](reviews/20261107-pipeline-fixes.md) maps each fix to its regression evidence.

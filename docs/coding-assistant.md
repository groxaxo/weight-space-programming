# Qwen3.5-4B coding-assistant workflow

This track adapts the official Qwen3.5-4B checkpoint for coding and structured tool use. It produces a separate LoRA adapter and preserves the base files. A completed training run is a candidate; deployment and broad coding reliability are separate questions assessed through the paired evaluations and real clients.

For the 7 November follow-up, the user selected the **nine historical Experimentos failures through OpenCode**, replacing the planned new all-164 run. Follow the separate [HE9 contract](he9-acceptance.md): fresh VM/client state, original frozen Base+Plus tests, 262,144 context, Q8 K/V and a strict matched gain before promotion. The all-164 utilities below retain their broader contract and historical evidence; HE9 qualification does not pass that gate.

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

Use `--phase smoke` for two optimizer steps, or `--phase evaluate` for the model tests. Evaluation accepts `--adapter /absolute/path/to/training/adapter`, `--clients`, `--projects`, and `--humaneval`. `--projects` runs both contained two-module file/terminal trials sequentially after file-edit clients and before HumanEval. Client quality failures are retained without suppressing the remaining benchmark. A smoke adapter is explicitly refused as a serving candidate.

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

For the complete local acceptance gate, also pass `--baseline-humaneval`, `--candidate-humaneval`, `--clients` (the candidate's `acceptance.json`), and `--training` (its `training-result.json`) to `compare_coder.py`. This requires all 164 paired tasks, a HumanEval solved count strictly higher than the base, both installed clients making edits that pass independent tests, and a finite improvement in held-out validation loss. The user strengthened this requirement on 7 November 2026: a tie no longer qualifies. The native Q8 gate also requires a strict win within its own paired profile. A missing or mismatched report fails closed. Weight/configuration checksums and the training-result fingerprint are mandatory. Each successful client result must also carry a verified request attestation from the current proxy. Historical schema-1 reports are rejected by the current gate; use their frozen source to inspect historical decisions, and regenerate baseline/candidate evidence for a new qualification run. Do not add identity fields to old reports retroactively.

## Corrective tuning with public replay

The optional `configs/qwen35-4b-corrective-mlp.json` recipe trains a rank-8 adapter only on language-layer MLP linear modules. Attention, vision and the output head remain frozen. Existing recipes retain the default `all-language` scope. Reduced scope is an experiment whose quality must be measured; it does not establish preservation of the base's capabilities.

Build the corrective corpus on .54 using the existing authenticated public import:

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python scripts/build_corrective_corpus.py \
  --public-corpus assets/coder-corpus-v1 \
  --output assets/coder-corpus-v3-corrective \
  --config configs/qwen35-4b-corrective-mlp.json \
  --model-proof reports/model-inputs.json
```

The producer authenticates replay files against their prepared manifest, retains the original public task split, adds 128 authored coding and 160 authored tool conversations, and checks every authored coding target through executable assertions in the pinned Docker grader. Whole conversations must fit the native tokenizer's training budget. All variants of an authored semantic family share a task identity and stay together in one split. The recorded V3 dataset has 832 training and 96 validation conversations.

The generator reads no HumanEval or client-fixture files. Known development failures inform its requirements, including empty ranges, concise code, escaped text, observed failure summaries and grounded file edits. Development scores are therefore tuning evidence. Family separation and a prompt-only lexical audit cannot establish semantic disjointness or exclude public/pretraining contamination. Preserve `build.json`, source identities, dataset hashes and the exact generator with any exported adapter. Keep original failures and rejected candidates when selecting a checkpoint. A complete training epoch or lower held-out loss does not authorize promotion.

The additional `scripts/build_agent_flow_corpus.py` producer adds 256 authored read/edit/test/finish trajectories to that authenticated corpus. Eight unrelated helper families use reduced OpenCode/Hermes tool schemas; every argument name is supported by the recorded installed APIs. These are synthetic trajectories, with simulated file-tool feedback and actual separately executed Docker unittest results. Both the broken starter and the corrected whole-file implementation are executed. Raw controls are archived; only their workspace paths are rebound and their displayed stderr is bounded when constructing a conversation. No client fixture or HE9 answer is imported.

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python scripts/build_agent_flow_corpus.py \
  --project /home/op/work/weight-space-programming-20261106 \
  --public-corpus assets/coder-corpus-v1 \
  --existing-corpus assets/coder-corpus-v3-corrective \
  --output assets/coder-corpus-v4-agent-flows \
  --config configs/qwen35-4b-agent-flows.json \
  --model-proof reports/model-inputs.json
```

The recorded flow corpus has 992 training and 192 validation conversations. Every client, repair state and variant of one helper stays in the same split; three new helper families are held out. The 256 additional conversations fit within 1,338 tokens, below the 1,536-token training budget. The optional flow recipe changes adaptation to all language-layer linear modules at rank 8/alpha 16, one epoch and learning rate 3e-6. This is a joint data/scope/rate experiment, so a downstream change cannot be attributed to one of those factors alone. Its checkpoint is selected by held-out loss before model testing.

Rebuilding the corpus executes fresh controls, whose observed timing text can change the data hash. Use the archived prepared files and their authenticated manifest to reproduce the exact recorded training inputs. Both source producers reject an existing output directory and retain the generated data, source identity and control record. The flow producer also retains its exact generator. The maximum training sequence length is separate from the measured 262,144-token native inference context.

For continuation from an existing adapter, pass `--init-training /absolute/path/to/completed/training` to `ops/run54_gpu_phase.py --phase train` or directly to `scripts/train_coder.py`. This starts a new optimizer with the recorded learning rate. It authenticates the parent's frozen recipe and adapter files, requires the same pinned base, rank, alpha and target scope, checks the constructed model's exact target set, and verifies every loaded FP32 factor against the saved tensor before the memory probe. `initial-adapter.json` records that equality and the parent identities; `run-manifest.json` records the parent before model loading. Incomplete, smoke, altered or incompatible parents are rejected. Optimizer and scheduler state are not resumed.

PEFT can save a long explicit target list as compact module suffixes. The verifier checks that summary against the recorded full targets, then constructs the adapter from the current model's full target list and requires exact saved tensor keys, shapes, dtypes and values. The compact configuration is never used to broaden the injected target set.

The `configs/qwen35-4b-agent-continuation.json` recipe retains the MLP-only rank-8 scope and uses learning rate 2e-6 for one new epoch. Its `scripts/build_agent_continuation_corpus.py` producer replays the authenticated public/corrective/flow corpus and adds exact `edit`/`patch` replacements plus once-only failure/success observations. It reuses the eight toy families' separately recorded Docker outcomes; file API responses remain simulated. Every new state and client retains the previous family identity and split. The producer requires the recorded parent manifest, authenticates its control record, tokenizes every full new conversation without truncation, and rejects an existing output. These development-informed inputs exclude HE9 and client-fixture answers. A continuation still requires the complete frozen HE9 and real-client promotion gate.

Generated Python executes only in the pinned Docker image with networking disabled, a read-only root filesystem, an unprivileged user, bounded memory/CPU/processes, and a read-only fixture mount. Sandbox self-tests run before scoring. This isolates generated code for these evaluations; it is not a general policy for unrestricted agent shell commands.

The checker requires a completion exit marker after all assertions, so an ordinary `exit(0)` cannot bypass the tests and produce a passing score. The Docker integration tests cover correct code, incorrect code, `SystemExit(0)`, and `os._exit(0)`. Run them with `WSP_TEST_DOCKER_IMAGE` set to the pinned image; the CPU CI workflow sets this automatically. This completion check is for grading, rather than a guarantee against deliberate grader manipulation.

If only sandbox scoring changes during a long HumanEval run, `rescore_humaneval.py` can regrade every saved output without regeneration. It requires the original frozen runner source, verifies its hash, verifies the generation client and code extractor are unchanged, authenticates the exact dataset/tests, and records the parent report hash and both runner versions. Generation changes require a new run.

## OpenCode and Hermes

After a client exits or reaches its deadline, the harness drains its already-issued provider requests before checking model discovery and attribution. This finalization wait does not extend the actor budget or change a timeout into success. Project trials stop only their own uniquely labeled containers before hashing files and running independent checks; scoped cleanup and provider draining also run on error paths. A changed client harness requires fresh paired compatibility measurements. Keep older missing-attribution or interrupted trials as incomplete evidence rather than copying proof from a newer run.

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

## Experimental HE9 community release

[Qwen3.5-4B-WSP-HE9-MLP](https://huggingface.co/groxaxo/Qwen3.5-4B-WSP-HE9-MLP)
shares the strongest measured adapter from the selected historical nine-task
OpenCode experiment. It improves Base 6/9 → 7/9 and Base+Plus 3/9 → 5/9.
Its complete assistant qualification fails: development drops 12/14 → 11/14
and the Hermes two-module project fails. The official base remains the selected
default. Publication is a research release, separate from promotion.

The model repository includes unchanged PEFT factors and an unmerged native F32
LoRA, a pinned public base locator, loading examples, method, raw evaluation,
artifact hashes, tested source, license and corpus attribution. Only the portable
configuration's base locator/revision differ from the measured configuration;
both hashes and the original configuration are retained. A CPU shape-only-base
standard-PEFT control verifies all 192 factors load exactly; it does not run
inference or establish quality for a different precision/runtime.

Follow the model card's pinned Qwen3_5ForConditionalGeneration/PEFT example for
Python use. The measured native profile uses the existing authenticated export
and conversion workflow, this project's chat/tool frontend, 262,144-token
context and Q8 K/V cache. Conversion manifests bind measured local paths and
are not portable installation files. The supervisor is specific to .54; adapt
GPU/service ownership deliberately on another machine. Examples define separate
OpenCode/Hermes profiles without changing existing providers. Full allocation
does not establish recall or coding quality at the full window.

The [comprehensive HE9 report](experiments/qwen35-4b-he9-20261107.md) and
[integration findings](reviews/20261107-he9-integration-findings.md) retain the
failed flow/continuation experiments, original incomplete client record and
all regression evidence. The later continuation has lower held-out loss but
only 9/14 development success; further tuning was stopped at the user's request
to close out and share current work. Next work should address family-disjoint
executable coverage, truthful tool-failure handling, permitted-command planning
and cross-module Hermes correctness, followed by repeated paired trials and
unseen repository tasks. No gate is relaxed for this release.

The historical 1,024-token omitted-limit defect is now resolved. The frontend
defaults to the configured `--max-output` allowance, and the new paired native
client profile uses `--max-context 262144 --max-output 28192 --cache-type-k q8_0
--cache-type-v q8_0`. Explicit smaller request budgets remain honored.

The paired 28,192-token client rerun records base **3/4** and released MLP **3/4** passes. All actual clients have complete request attribution: **True**. The omitted-limit fallback is fixed, native output allowances are verified, and the original HE9 measurements retain their separate 8,192-token identity. No additional training or model promotion is performed.

See the [complete follow-up](experiments/hermes-output28192-20261107.md) and [declared protocol](experiments/hermes-output-28192-protocol-20261107.md). The historical malformed continuation calls remain in their original failed record. Current budget proof comes from actual native settings, with main Hermes tool calls and auxiliary requests both accounted for.

# Experimentos HE9 replay contract

The user selected the **nine historical coding failures replayed through OpenCode** from Experimentos. This replaces the planned new all-164 HumanEval run for this experiment. Existing broader acceptance utilities retain their own contract; a selected HE9 result does not pass that broader gate.

## Frozen tests

Source: private `groxaxo/experimentos`, inspected revision `70e4eabb39c76287f67ce618d27f1c4f7d2ea2d0`, `mimo9b/opencode-he9-2026-10-19/`. Originals were recovered read-only from `/home/op/experimentos/mimo-he9-corrected-20261019T004652Z` on `.51`. No `.51` inference job is involved. Private archive files are supplied separately and are not redistributed in Git.

| Original file | SHA-256 |
|---|---|
| `grade_worker.py` | `fedd6addd2e03df62f00185ac0f0beb4123f34bc99e731b753076a51bf5392a5` |
| `grader-cases-private.jsonl` | `a9da6444d2142606e0d7ca7f81971f65f595bbb55370e322fc8480c578b18b4d` |
| `suite-selected.json` | `e1d7d6e8b79b82ae9649a05db506d2ce40f89401de0067429b5faa6956ea3add` |

| Task | Frozen Plus inputs |
|---|---:|
| HumanEval/1 | 83 |
| HumanEval/6 | 66 |
| HumanEval/32 | 788 |
| HumanEval/39 | 2 |
| HumanEval/76 | 897 |
| HumanEval/99 | 710 |
| HumanEval/101 | 1,000 |
| HumanEval/132 | 1,000 |
| HumanEval/141 | 1,000 |
| **Total** | **5,546** |

All nine public starter files must match the original `problem_sha256` byte for byte. Hidden inputs, original base-test code, canonical implementations and the grader stay outside the actor workspace and corrective training corpus. The unchanged **`reconstructed-base-plus-v1`** script runs the archived base tests followed by the frozen Plus inputs. It is not a certified official EvalPlus implementation. All nine references must pass every base and Plus check in the pinned grader before inference.

HumanEval/32's canonical-root comparison can reject a different mathematically valid root. Preserve that frozen verdict and report any supplementary root audit separately. Historical Qwen single-prompt failures and MiMo scores provide context, not matched measurements of the new Qwen4B base.

## Matched runtime

| Setting | Required value |
|---|---|
| GPU host | `.54`, RTX 3060 12 GB, one owned GPU lease |
| Client/grader | New dedicated Ubuntu VM on `.54` |
| Qwen3.5-4B revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Model format | The same authenticated Q8_0 base in both arms; candidate uses separate unmerged FP32 LoRA, scale 1 |
| Context | **262,144**, the authenticated checkpoint's supported maximum |
| OpenCode input/output | **253,952 / 8,192** tokens |
| KV | **Q8_0 K and Q8_0 V**, flash attention enabled |
| Sampling | Non-thinking; temperature **0.7**, top-p **0.8**, top-k **20**, min-p **0**, presence penalty **1.5**, repetition penalty **1**, frequency penalty **0**, fixed seed **20261106** |
| OpenCode | **1.19.4** |
| OpenCode ELF SHA | `538fcb1b6408c8d4608822acee19f7ff5b3a3eb3383c0633c4f06170857312a7` |
| Actor/grader Python | **3.11.17** |
| Actor budget | **1,200 seconds per task** |
| Grader | Separate network-disabled, read-only container; 320-second outer limit |

The historical actor used OpenCode 1.18.35 and a 32,768-token context; its old images are no longer installed. This replay uses a new pinned image, newer installed client and the user-requested maximum context/Q8 KV. Record these runtime differences while preserving exact task and grader bytes.

The first thinking-enabled attempt and two diagnostic reproductions are retained as unscored failure evidence. Raw tokens show the base emitting a tool call inside an unclosed thinking block and then ending the turn. The strict parser continues to reject that output. The complete matched run uses the non-thinking chat-template mode used by the corrective training recipe, identically for base and adapter. Thinking-mode reliability remains unqualified; no failed attempt is relabeled as a passing score.

A subsequent greedy non-thinking attempt repeatedly reran an unchanged failing command despite receiving its real output. It is preserved as an interrupted development attempt. The complete pair uses the instruct sampling settings in the [pinned model card](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md), identically for base and adapter. The publisher identifies presence penalty as a repetition control. This is a runtime change from the historical greedy replay; task/grader bytes and the original 8,192-token output cap remain fixed. One seeded attempt per task does not establish statistical superiority.

`--sampling-profile qwen35-instruct` binds the settings into serving identity, supplies omitted client defaults, rejects contradictory explicit values, forwards all settings including the seed, and verifies the native engine's reported effective values. The client proxy additionally checks effective sampling on each response. The previous native request builder hardcoded top-k/presence values and did not forward the request seed; HTTP regressions cover this discrepancy. Standalone evaluators and client configurations now follow advertised model limits/settings instead of forcing greedy/16K defaults. Historical greedy workflows retain an explicit separate profile.

Create the VM from a verified pristine cloud image with a new disk overlay. Before installation, record creation time, cloud-image hash, VM identity, machine-ID hash, and absence of prior OpenCode binary/state, Hermes state and harness workspace. Preserve personal VMs. Each actor receives fresh tmpfs HOME/XDG state, a public-only `/work`, two read-only harness scripts and a read-only Unix-socket mount. It has no external network, Docker socket, host home, private grader or previous session. File/terminal tools operate inside the container. MCP, external skills/plugins, web and delegated task tools are disabled for this controlled replay.

The model stays on host loopback. A loopback-only SSH reverse tunnel reaches the VM; a per-task attribution proxy and Unix relay reach the network-disabled actor. Validate the exact model identity and server instance on every actual JSON/SSE completion. Capability probes cannot count as inference. Actual container inspection is required.

Controlled runs opt into private request/token traces before parsing. These preserve malformed model output for diagnosis without relaxing validation. The native-port readiness probe permits TCP `TIME_WAIT` reuse while still refusing a live listener. A regression exercises both real socket states.

At the deadline, stop only the owned actor, preserve and grade its saved artifact, and report agent completion separately. Drain issued requests before attribution. A disconnect after a validated response is delivery failure, not model-identity drift. Stop that task's frontend/native processes and restore Bonsai before the next task. Each next task starts a new model instance. Infrastructure/API/grader errors remain unscored and invalidate qualification.

## Qualification

Predeclared requirements for the matched pair:

1. Exactly the same nine distinct, fully scored tasks in each arm, with authenticated tests/starters, reference controls, pinned actor, source, runtime, context and Q8 K/V settings.
2. **Strictly more Base+Plus tasks solved** than the new Qwen4B base. A tie fails.
3. No previously solved Base+Plus task or original base-test task is lost.
4. The number of completed OpenCode actors does not decrease. Correct saved code at a deadline remains a score but is not completed agent work.
5. Independent OpenCode/Hermes development compatibility must pass before local promotion. Lower training loss alone never qualifies an adapter.

`scripts/compare_he9.py` returns `he9_qualified_pending_compatibility` or `not_promoted`. It always records `adapter_promoted: false`; endpoint selection is separate. Keep failed attempts/candidates. Never replace failures with retries, rewrite old identities or import canonical HE9 answers into training. A protocol change requires a new complete matched pair.

This small, publicly known failure-selected slice is diagnostic tuning evidence. Even 9/9 cannot establish broad repository competence, uncontaminated generalization, all-164 HumanEval accuracy, frontier equivalence or perfection.

## Operation

Build [the actor image](../ops/he9/Dockerfile) using only the authenticated ELF; record its immutable image ID. Keep the private files and original reference self-tests outside actor workspaces. Setup metadata records all nine reference results, image IDs, source manifest and versions. Verify the actual CLI path with a synthetic provider, without Qwen inference or benchmark answers:

```bash
/home/runner/he9-env/bin/python ops/verify_he9_actor.py \
  --runtime /home/runner/wsp-he9/setup/runtime.json \
  --output /home/runner/wsp-he9/controls/synthetic-unique
```

Run each arm on `.54` in a bounded systemd unit with `KillMode=control-group`, output-specific restoration and no competing GPU phase:

```bash
.venv-train/bin/python -u ops/replay_he9_vm54.py \
  --conversion runs/native-conversion/conversion.json --profile base \
  --output runs/he9-base-unique \
  --control /home/op/work/wsp-he9-vm-control-20261107 \
  --guest 192.168.122.207 \
  --guest-output /home/runner/wsp-he9/runs/he9-base-unique
```

Use identical source/setup/runtime with the candidate's unmerged conversion profile for the second arm. Each path must be new. Progress and final reports are published without overwriting prior evidence:

```bash
.venv-train/bin/python scripts/compare_he9.py \
  --baseline runs/he9-base-unique/replay.json \
  --candidate runs/he9-candidate-unique/replay.json \
  --output reports/he9-comparison-unique.json
```

Preserve host commands/logs, guest transcripts, saved candidates, grader logs/results, reference controls, source hashes, VM creation/virgin proof and comparison. SSH private keys and personal profiles stay out of reports, archives and Git. Clear HF cache only when necessary, preserving needed authenticated inputs and Bonsai/Moss runtimes.

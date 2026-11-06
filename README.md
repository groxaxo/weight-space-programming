# Weight-Space Programming Lab

A verified reproduction of the published **Qwen3.5-0.8B p50c weight repair**, plus a pinned **Qwen3.5-4B coding-assistant adaptation and evaluation track**. Both keep the original base checkpoint unchanged.

**Reconstruction:** pinned downloads, full input authentication, sparse BF16 reconstruction, non-overwriting publication, output and provenance verification, and CPU regression tests. A real 1.75 GB checkpoint has been reconstructed and checked on the Ubuntu 12700. `verify-output` now also authenticates the reconstruction record against the actual files and trusted lock.

**Coding track:** authenticated official model snapshots, licensed public corpus imports, complete task-disjoint conversations, assistant-only loss masks, BF16 LoRA / NF4 QLoRA, separate validation and checkpoint selection, bounded OpenAI-compatible serving, executable coding and multi-turn tool checks, all 164 HumanEval tasks, and isolated OpenCode/Hermes file-edit acceptance checks. Generated Python is graded in a digest-pinned Docker sandbox with no network and a read-only root filesystem. See [the coding-assistant workflow](docs/coding-assistant.md).

**Scope:** this does not implement autonomous Sol/Astra orchestration, activation extraction, or a general sandbox for unrestricted agents. Model quality is established by the actual paired reports and client results, not by successful training or reconstruction alone. The compact development suite is not evidence of frontier-model equivalence.

The current coding experiment runs on **`100.85.200.54`**, an RTX 3060 with 12 GB VRAM. Its supervisor temporarily pauses the existing Bonsai service only when explicitly authorized and restores it at the end of each GPU phase, including failures. Moss and Bonsai models/runtimes are preserved.

## Current measured findings

**No fine-tuned adapter passed the complete acceptance gate.** The official base
remains the default: 6/8 authored coding tasks, 6/6 tool scenarios and 128/164
HumanEval tasks (78.05%). Both real OpenCode/Hermes clients passed the initial
single-module file-edit fixture. Lower-rate adapters improved the small coding
suite to 7/8 but scored 127/164 and 126/164 HumanEval, so neither was promoted.

The [comprehensive experiment report](docs/experiments/qwen35-4b-20261107.md) records
the training recipe, all rejected profiles, harder project trials, the corrected
Hermes Docker configuration, grader controls, storage cleanup and service
restoration. [Machine-readable findings](docs/experiments/qwen35-4b-20261107.json)
preserve identities and per-case changes. Successful training or a small edit
fixture does not establish broad coding reliability or frontier-model parity.

The [pipeline review and fixes](docs/reviews/20261107-pipeline-fixes.md) close all nine
reviewed discrepancies, including GPU ownership, mandatory artifact binding, atomic
checkpoint export, literal tool markup and attribution of actual client requests.
Serving identity schema 2 requires fresh paired evidence; the historical quality
measurements above remain attached to their original source.

The source regression suite contains 136 tests. Use the independent CPU setup
below; optional training-library checks skip in that environment. Full sandbox
checks additionally require Docker and `WSP_TEST_DOCKER_IMAGE`, as configured
in the included GitHub Actions workflow.

## Quick start — Linux, Python 3.12

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/pipeline.py bootstrap
.venv/bin/python scripts/pipeline.py download
.venv/bin/python scripts/pipeline.py verify
.venv/bin/python scripts/pipeline.py rebuild
.venv/bin/python scripts/pipeline.py verify-output
```

`bootstrap` fetches only the two pinned edit artifacts, not an executable upstream checkout. `download` independently fetches the locked base snapshot. Existing files are reused only after size and SHA-256 verification. Incorrect files are **not silently replaced**. Downloads are streamed to temporary files and published without clobbering existing paths; interrupted downloads restart that file rather than resuming it.

The default paths are `assets/p50c`, `models/base`, and `models/rebuilt`. `preflight` is an alias for the full input/tensor verification performed by `verify`, not merely a presence check. Rebuild and verification are offline after the inputs exist. They require no GPU, frontier-model calls, API keys, or authenticated services.

### Reuse the existing model on the 12700

The review branch is checked out at `/home/op/work/weight-space-programming-review-20261106`. The original checkout remains unchanged on `master`.

```bash
cd /home/op/work/weight-space-programming-review-20261106
PY=/home/op/experiments/weight-space-programming/.venv/bin/python
BASE=/home/op/experiments/weight-space-programming/models/base
OUT=/home/op/experiments/weight-space-programming/models/rebuilt

"$PY" scripts/pipeline.py verify --base "$BASE"
"$PY" scripts/pipeline.py verify-output --base "$BASE" --output "$OUT"
```

The verified output already exists at `$OUT`. A second rebuild to that directory deliberately fails; use a new output directory for another run. `--base` accepts either the model directory or an exact locked shard path. **`--output` is now a checkpoint directory**, not the old single `.safetensors` filename. The checkpoint index resolves the real shard name automatically.

## Integrity and publication guarantees

`reproduction.lock.json` pins the model revision, research revision, every required model file's size/SHA-256, the edit archive and manifest, and the ordered set of edited tensors. Base model hashes were cross-checked with Hugging Face's immutable-revision LFS SHA-256 values or Git blob IDs. This lock and the code in this repository are trusted inputs: changing both is outside the integrity threat model.

The implementation verifies the entire base, including untouched weights and sidecar files; applies only the published sparse BF16 edit; verifies the published pre-edit/post-edit tensor hashes; and checks every byte outside the edited tensor spans. It copies the full checkpoint into a private staging directory, rechecks those copies, records provenance inside that directory, and publishes using Linux `renameat2(RENAME_NOREPLACE)`. Existing files, directories, empty directories, and dangling symlinks cannot be replaced by the publication operation. Normal failures remove only the current invocation's staging directory.

Guards use explicit exceptions, not `assert`, and remain active under `-O`, `-OO`, and `PYTHONOPTIMIZE`. Upstream Python is never executed, and NumPy archives are loaded with `allow_pickle=False`. Unsupported edit forms are rejected rather than silently generalized.

A killed process or machine crash may leave a hidden `.NAME.building-*` directory, but not a partial checkpoint at the final destination. Inspect stale staging directories before removing them; another run may still own one. Download/rebuild operations should use private directories controlled by the operator. This is **not** a defense against an attacker who can rewrite the process's files, trusted lock, code, or directory tree while it runs. Linux no-replace publication is mandatory; there is no unsafe portable fallback.

## Measured reconstruction

| Check | Observed result |
|---|---|
| Full base checkpoint and 12 sidecar files | All locked hashes matched |
| Layer 15 MLP tensor | Published edited SHA-256 matched; 65,260 BF16 entries differ |
| Layer 19 MLP tensor | Published edited SHA-256 matched; 64,928 BF16 entries differ |
| Total changed BF16 entries | 130,188 |
| All remaining weight/header bytes and sidecars | Preserved |
| CPU regression suite on the 12700 | 40 passed |
| GPU inference / MATH-500 performance | Not run |

The often-quoted 131,072 figure counts selected weight positions. The published manifest counts 130,188 positions whose BF16 bits actually change; some touched entries round back to their original value. Both edited tensor hashes match the published artifact. The whole output file's hash is **not** claimed to equal a separately repackaged Hugging Face release, since serialization/header layout may differ.

See [review comparison](docs/reviews/20261106-comparison.md), [Sol's original static findings](docs/reviews/sol-6.1-static-review-20261106.md), and [measured validation evidence](docs/reviews/validation-20261106.json).

## Research provenance

- Research: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`, `exp8_math/runs/p50c/edit`.
- Base: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), revision `2fc06364715b967f1860aea9cf38778875588b17`.
- Format: [Safetensors specification](https://github.com/safetensors/safetensors#format).

This project does not redistribute checkpoint or edit binaries in Git. Their upstream terms apply. Local testing is independent of GitHub Actions. Successful weight reconstruction establishes artifact integrity, not a new discovery result or reproduction of benchmark scores. Keep practice data separate from held-out evaluation when adding the research-agent stages.

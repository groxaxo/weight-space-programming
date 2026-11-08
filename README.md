# Weight-Space Programming Lab

The [unified forward-only lab](docs/unified-lab.md) combines the compatible logic
from the six reviewed projects behind one CLI: `python scripts/wsp_lab.py`.
It provides frozen protocols, fixed BF16 activation controls, the shared `.54`
sandbox, literal diagnostic grading, paired development/locality metrics, sparse
patch-format inspection and historical replay. Fitted probes, target optimizers
and LoRA workflows are excluded. Integration qualification is engineering evidence;
no new 4B repair or independent model-quality result is claimed.

A verified reconstruction of the published **Qwen3.5-0.8B p50c weight repair**, plus an active **Qwen3.5-4B direct-weight-edit research track**. The original base checkpoints and retained historical artifacts are preserved. **A successful 4B direct repair has not yet been established.**

**Reconstruction:** pinned downloads, full input authentication, sparse BF16 reconstruction, non-overwriting publication, output and provenance verification, and CPU regression tests. A real 1.75 GB checkpoint has been reconstructed and checked on the Ubuntu 12700. `verify-output` now also authenticates the reconstruction record against the actual files and trusted lock.

**Active 4B method:** the [pinned upstream exp8 constraints](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md) require forward-only diagnosis and direct tensor interventions. Training, LoRA/QLoRA, PEFT, backward/autograd updates, optimizers, loss fitting and merging trained adapters are excluded. The new [CPU patch contract](docs/direct-weight-editing.md) pins the official BF16 4B parent, authenticates all runtime inputs and validates sparse uint16 BF16 replacements against exact before/after hashes, counts and limits. It writes no full checkpoint. Bounded practice observation and resource admission are separate first-stage tools; a causal repair, independent paired acceptance and native deployment remain unproven. See [active agent instructions](AGENTS.md) and [method policy](research-policy.json).

The [8 November direct-weight study](docs/experiments/direct-weight-20261108.md)
records the provenance audit, exact-base reproduction, unchanged-parameter
semantic diagnostics, finite activation controls, preservation and negative
results. [Frozen practice records](docs/experiments/direct-weight-20261108/)
are exposed research evidence, not untouched evaluation data.
The finite 14-forward activation study found a reproducible effect in layers
16–23 but failed its frozen reverse joint-margin check. No weight candidate was
justified, independent paired model quality remains `NOT_RUN`, and the baseline
remains the retained reference.

The [follow-up harness repair report](docs/reviews/20261108-direct-harness-fixes.md)
records eight fixes for parent/input authentication, concurrent admission,
interrupted observations, completion, donor gates, host evidence and exact-cap
EOS handling. New worker acceptance requires source-bound evidence schema 2.
These engineering repairs preserve the negative scientific result above; they
introduce no weight candidate or independent quality claim.

A [new tool-finalization hypothesis](docs/plans/direct-tool-finalization-cycle2-20261108.md)
is separately preregistered: test whether the unchanged BF16 model repeats a
completed failed-test call before investigating a sparse persistent-action write.
The initial gate uses two fresh full episodes and at most four CPU generation
invocations; it permits no weight edits. A separate user-selected
[Sol max custodian](docs/plans/direct-cycle2-independent-acceptance-20261108.md)
defines later independent paired acceptance. No candidate or new quality result
is established at this protocol freeze.

The first tool tranche ended `STOP_harness`: the CPU observer continued beyond
assistant-turn EOS. The [F09 repair](docs/reviews/20261108-direct-chat-stop-repair.md)
passes the authenticated EOS union explicitly and verifies the first stop from
raw tokens. Original evidence remains unchanged. One separately reviewed,
fully charged technical replay uses fresh source and capsules, with every
scientific criterion unchanged; no weight candidate or quality pass follows
from repairing this harness.

The corrected tool replay has now reproduced the duplicate call in both fresh
BF16 episodes, with exact greedy token replay and all 426 parameters unchanged.
The five fixed semantic controls in the
[Cycle 2 report](docs/experiments/direct-tool-finalization-cycle2-20261108.md)
reject the proposed nonzero-result-specific mechanism: the success-result
control also repeats the tool call. No localization, weight candidate or
independent model-quality result is admitted by this cycle.
The [control integration review](docs/reviews/20261108-direct-tool-specificity-integration.md)
records the authenticated historical evidence boundary, three-position capture,
CPU sandbox, cumulative accounting and regression checks. The original BF16
checkpoint remains the reference; no direct-edit model has been promoted.

The [comparison with six related GitHub repositories](docs/reviews/20261108-related-direct-editing-repositories.md)
checks these method boundaries against actual source. Forward intervention and
matched evaluation are useful references; ROME/MEMIT target optimization and a
popular one-neuron framework's fitted-probe/LoRA paths do not satisfy this
project's no-gradient contract. The review also separates our local sparse-edit
hypothesis from upstream requirements and labels the historical client examples.

**Historical coding track:** the preserved LoRA/QLoRA work contains authenticated model snapshots, licensed corpora, assistant-only loss masks, adapter selection, serving, executable coding/tool checks, HumanEval and OpenCode/Hermes fixtures. Those artifacts and results remain useful records, but they are not implementations of the active no-training method. Legacy training, supervised-corpus preparation and trained-derivative CLI entry points are guarded by `scripts/method_policy.py`. See [the historical coding-assistant workflow](docs/coding-assistant.md).

**Scope:** the original `0be1a86` wrapper reconstructs an existing 0.8B patch; it does not discover a repair. Its selected coordinates are never transferred to 4B. Model quality requires matched reports and independent evaluation, beyond successful reconstruction or a valid numeric patch. The exposed HE9/development/client fixtures are regressions, not a fresh untouched holdout. No result here establishes frontier-model equivalence.

The research host is **`100.85.200.54`**, with an RTX 3060 and 12 GB VRAM. New work requires live CPU/RAM/disk/GPU admission. Do not pause, restart, reroute or reconfigure Spark Fable Runtime, Bonsai, Moss, other jobs or the separate evaluation. Queue unavailable resources and preserve all artifacts; no cleanup is authorized. Sparse candidates avoid a second 9.32 GB BF16 checkpoint copy.

## Historical LoRA findings — preserved, not direct-edit results

**Semantic repair round (8 November 2026):** A new MLP continuation lowered held-out loss from 0.2627 to 0.1165 but scored **10/14 development cases and 2/4 actual clients**, against a fresh base **12/14 and 3/4**. It is rejected: `bounded_chunks` still fails, escaping/version comparison regress, OpenCode's project times out, and Hermes's project fails. All eight client trials have valid request attribution; all 177 client requests receive the 28,192 native allowance. See [the full negative result](docs/experiments/semantic-repair-20261108-round1.md), [raw paired evidence](docs/experiments/semantic-repair-20261108-round1-paired.json) and the historical [formatted-data follow-up protocol](docs/experiments/formatted-repair-protocol-20261108.md). That training follow-up was stopped after logged step 22/172 without saving a new checkpoint; it is not the active plan. Released Hugging Face weights retain their historical trained identity.

**28,192-token client follow-up:** The paired 28,192-token client rerun records base **3/4** and released MLP **3/4** passes. All actual clients have complete request attribution: **True**. The omitted-limit fallback is fixed, native output allowances are verified, and the original HE9 measurements retain their separate 8,192-token identity. No additional training or model promotion is performed. See [paired results and budget proof](docs/experiments/hermes-output28192-20261107.md).

The latest selected **Experimentos HE9** replay uses actual OpenCode in a fresh
VM on `.54`, full 262,144-token context and Q8 K/V cache. The matched base scores
**6/9 Base and 3/9 Base+Plus**; the MLP adapter scores **7/9 and 5/9**, respectively,
with 7/9 completed actors in each arm. These are the original 8,192-output-cap measurements. Both newly solved tasks and the remaining
partial-result regressions are retained. The original complete qualification rejects the
MLP adapter: development success falls from 12/14 to 11/14, and the Hermes project
trial fails. A further all-language adapter scores 6/9 Base and 4/9 Base+Plus,
solving task 132 while losing base-solved task 76. An authenticated MLP
continuation scores 6/9 Base, 4/9 Base+Plus and 9/14 development cases despite
lower held-out loss. **No adapter has been promoted**.

The best MLP checkpoint is shared as an **experimental community release**:
[Qwen3.5-4B-WSP-HE9-MLP on Hugging Face](https://huggingface.co/groxaxo/Qwen3.5-4B-WSP-HE9-MLP).
It includes the unchanged PEFT/native LoRA weights, portable configuration,
model card, method, all candidate results, tested code and attribution. This is
a partial research success and a failed complete assistant qualification.
Publication does not change the default provider or relax the promotion gate.
The [complete HE9 report](docs/experiments/qwen35-4b-he9-20261107.md),
[machine-readable evidence](docs/experiments/qwen35-4b-he9-20261107.json),
[seven additional integration fixes](docs/reviews/20261107-he9-integration-findings.md)
and [HE9 contract](docs/he9-acceptance.md) record the gains, regressions, remaining
Hermes/tool failures, exact loading controls and recommended next work.

In the earlier, separately frozen broad experiment, the official base scored
6/8 authored coding tasks, 6/6 tool scenarios and 128/164
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

The source regression suite includes CPU integrity and protocol controls. Use the independent CPU setup
below; optional historical training-library checks skip in that environment. Full sandbox
checks additionally require Docker and `WSP_TEST_DOCKER_IMAGE`, as configured
in the included GitHub Actions workflow.

**GPU comparison and accounting (7 November 2026):** a targeted, single-GPU-per-arm MATH-500 runner compares original-thinking, repaired-thinking, and original-nonthinking through the existing GPUQ queue. The three jobs were subsequently cancelled after entering generation, with zero saved samples and no accuracy result. The token-accounting audit and interruption evidence are retained. See [protocol, job IDs, and operations](docs/THREE_GPU_COMPARISON.md).

**Research-agent limits:** no autonomous Sol/Astra orchestration, frontier token/spending budget controller, general unrestricted editor sandbox or independently accepted direct-edit model exists. The direct contract and bounded forward-only practice worker are engineering foundations. The historical coding and HE9 tracks provide isolated execution for their bounded fixtures. [Sol 6.1 Max produced a broader historical implementation plan](docs/plans/sol-6.1-max-20261107.md), with [invocation provenance](docs/plans/sol-plan-provenance-20261107.json); that plan is not evidence of execution.

## 0.8B reconstruction quick start — Linux, Python 3.12

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

Original project code and documentation use [Apache-2.0](LICENSE), with
[third-party notices](NOTICE). Imported models, datasets and research artifacts
retain their respective terms; see [corpus attribution](docs/coder-attribution.md).

## Actual token accounting and benchmark observations

[Actual measured token usage](docs/TOKEN_USAGE.md) separates recovered Sol session input, cached input, output, and reasoning subsets from local Qwen sample tokens. A read-only observer can refresh counts without another model call. It does not claim complete accounting for the main ChatGPT conversation or monetary billing.

The three RTX 3090 benchmark attempts loaded and entered generation on 7 November 2026, then received SIGINT before saving any samples. They are cancelled, with no accuracy result. [Interruption evidence and limitations](docs/usage/benchmark-interruption-20261107.md) records the unknown cancellation initiator and discarded-token accounting gap.

# Related GitHub repositories and method alignment

Reviewed on 8 November 2026, after Cycle 2 was merged into `main` and the
repository was made public. This is a primary-source code/document review of
six actual repositories at the revisions below. No third-party package was
installed, experiment executed, training started or model edited. File/blob
identities are recorded in the [source inventory](20261108-related-direct-editing-repositories.json).
Other projects' reported results were not independently reproduced here.

**Assessment:** the current no-training boundary, exact-base reproduction,
paired controls, preservation and refusal to promote a failed hypothesis are
appropriate. They do not establish that our proposed edit mechanism works.
Cycle 2 correctly stops at `STOP_semantic_specificity`: S repeats after success;
C and N produce identical fenced JSON. No inspected repository supplies evidence
that this specific Qwen3.5-4B failure can be repaired by our proposed columns.

| Repository / inspected revision | Relevant evidence | What applies here |
| --- | --- | --- |
| [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f), `4d09ba6` | [exp8 setup/rules](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md): direct edits, restricted editor/forward tools, separate practice and final evaluation; no gradients, training or optimizers. | This remains the provenance reference. Its 0.8B math experiment and omitted harness implementation are not a portable 4B coding-repair algorithm. Preserve model-specific diagnosis and separate final evaluation. |
| [kevinpl07/weight-patching](https://github.com/kevinpl07/weight-patching/tree/d7df0d9bd15afb952b1e0cb64a067806563aa8e3), `d7df0d9` | [Protocol](https://github.com/kevinpl07/weight-patching/blob/d7df0d9bd15afb952b1e0cb64a067806563aa8e3/docs/PROTOCOL.md): observable failure, matched random edit, prompt baseline, new items/wordings and locality guards. | Closest practical comparator. Its learned gate and training workflows do not meet our no-gradient contract; use the evaluation design as a reference, not its quick-start recipe. |
| [ndif-team/nnsight](https://github.com/ndif-team/nnsight/tree/4c7ebffab55313ed26f3690f1d93d3e420629dc3), `4c7ebff` | [Activation patching](https://github.com/ndif-team/nnsight/blob/4c7ebffab55313ed26f3690f1d93d3e420629dc3/docs/patterns/activation-patching.md): matched-position clean/corrupt transfer, reverse transfer and explicit baseline. | Supports forward intervention as a causal diagnostic. A rescued activation can identify an information carrier; it does not by itself justify changing permanent weights. No Qwen3.5 runtime compatibility was tested. |
| [stanfordnlp/pyvene](https://github.com/stanfordnlp/pyvene/tree/9e333904dcf9e597ca76170010d17f4d4580de8d), `9e33390` | [Intervention implementation](https://github.com/stanfordnlp/pyvene/blob/9e333904dcf9e597ca76170010d17f4d4580de8d/pyvene/models/interventions.py) separates collection/zero/vanilla operations from trainable interventions. | Fixed, nontrainable interventions are a useful conceptual reference. Trainable interventions are excluded. Generic model support is not evidence that its mappings handle our checkpoint correctly. |
| [kmeng01/rome](https://github.com/kmeng01/rome/tree/0874014cd9837e4365f3e6f3c71400ef11509e04), `0874014` | [compute_v.py](https://github.com/kmeng01/rome/blob/0874014cd9837e4365f3e6f3c71400ef11509e04/rome/compute_v.py) optimizes a target latent with Adam and backpropagation before constructing the edit. | Related model-editing literature, but this implementation violates our optimizer/backward prohibition. A rank-one final write does not make its discovery gradient-free. Factual association edits are not coding-agent validation. |
| [kmeng01/memit](https://github.com/kmeng01/memit/tree/80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b), `80426fd` | [compute_z.py](https://github.com/kmeng01/memit/blob/80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b/memit/compute_z.py) also optimizes target activations with Adam/backpropagation. | Its final algebraic weight update does not remove that fitting stage. It is excluded from the active method; no script or target-vector optimizer was imported. |

## The closest comparator has a material method difference

The weight-patching [README](https://github.com/kevinpl07/weight-patching/blob/d7df0d9bd15afb952b1e0cb64a067806563aa8e3/README.md)
describes a gate whose base-model parameters are not gradient-trained. However,
its [fit_gate implementation](https://github.com/kevinpl07/weight-patching/blob/d7df0d9bd15afb952b1e0cb64a067806563aa8e3/patchkit/methods.py#L87-L145)
calls a fitted classifier. The classifier's
[logreg_dir](https://github.com/kevinpl07/weight-patching/blob/d7df0d9bd15afb952b1e0cb64a067806563aa8e3/patchkit/llm.py#L153-L172)
enables gradients, optimizes probe parameters with LBFGS and calls backward.
Thus the complete gate-construction procedure fails our stronger prohibition,
even when the base model stays frozen. Its
[all command](https://github.com/kevinpl07/weight-patching/blob/d7df0d9bd15afb952b1e0cb64a067806563aa8e3/patchkit/run.py)
also includes a LoRA training arm. Neither command is an authorized shortcut here.
Its Qwen3-4B case is not our Qwen3.5-4B checkpoint, and no neuron indices transfer.

The useful comparison is experimental: require a real base failure, separate
practice/holdout, include random controls, and measure effects outside the target
task. Their protocol adds byte-identical ordinary outputs and chat-distribution
checks. Those can inform a future frozen locality protocol; they have not been
run here and must not become post-hoc thresholds on an exposed holdout.

## What our contract adds, and what remains unfinished

Our single-layer/eight-column deletion support, dominance thresholds, BF16 norm
ceilings and zero-regression rule are local experimental choices. They are not
universal requirements from upstream or proof that useful repairs must have this
form. The current sparse-write hypothesis was deliberately falsifiable and has
failed. Broader support or another algebraic intervention requires a separate
reviewed protocol; no new support is admitted by this literature review.

This is not a replication of exp8: editor, model size, task, sampler and harness
differ. We preserve its no-training principle and custody separation without
transferring its coordinates or claiming its measured gains. Its forward-pass
budget unit also differs from our conservative one-generation-per-worker count.
Our failure charging and terminal-gate rules remain the frozen local contract.

The independent Sol max custodian has only its public interface. No eligible
candidate exists; private cases and the independent final runner/checker have
not been created/executed. That runner/checker must be implemented and qualified
before a later final run. Independent quality, quantized conversion and native
OpenCode/Hermes acceptance remain `NOT_RUN`. None of the external projects fills
those gates for our repository automatically.

The next justified work is a new reviewed failure-isolation study, distinguishing
tool completion from generic wording/position effects and strict JSON formatting,
with representative development contexts and untouched final custody. It is not
another edit attempt under the failed Cycle 2 protocol. Training stays stopped.

## Documentation corrections from this review

The active `AGENTS.md` now explicitly distinguishes our hypothesis-specific support
limits from upstream rules and requires inspecting the complete construction
pipeline behind any training-free claim. A new [examples index](../../examples/README.md)
labels the old client configuration as historical: its 16384/2048 example limits
are not the current native acceptance profile and are not a direct-edit release.
The frozen CPU 512/96 diagnostic remains distinct from native 262144 context,
Q8 K/V and 28192 output. No profile was changed or falsely certified by this
documentation correction.

# Sol 6.1 Max static review

Target commit: `5367eaa`. Session: `01a1af02-5213-7461-b8c6-62bd19c11e10`.

This is the final review response recovered from the existing local session log. Sol did not execute tests, edit files, or review the subsequent remediation. Local path links below refer to the original reviewed checkout.

1. **P1 — Optimization disables safety and hash validation.** [pipeline.py:36](/home/op/experiments/weight-space-programming/scripts/pipeline.py:36) and [apply_edit.py:49](/home/op/experiments/weight-space-programming/upstream/lm_understand-logs/exp8_math/code/apply_edit.py:49) use `assert` for overwrite protection, tensor shape/dtype checks, and expected hashes. `PYTHONOPTIMIZE=1` removes these checks in both processes: rebuilding can overwrite the base, and mismatched tensors can still print “hash OK.” Use unconditional exceptions.

2. **P1 — The published edit has no trusted integrity anchor.** [pipeline.py:24](/home/op/experiments/weight-space-programming/scripts/pipeline.py:24) checks file existence and a nonempty manifest; archive hashes are merely printed. Removing one tensor from the manifest produces an incomplete edit that still passes every remaining tensor check. The wrapper also executes the unchecked upstream Python file at [pipeline.py:41](/home/op/experiments/weight-space-programming/scripts/pipeline.py:41). Pin and verify the upstream source, manifest, archive, and expected tensor set.

3. **P2 — An incompatible base can pass validation.** [apply_edit.py:47](/home/op/experiments/weight-space-programming/upstream/lm_understand-logs/exp8_math/code/apply_edit.py:47) validates only the two edited tensors. Corruption or changes to another tensor survive reconstruction, while [pipeline.py:42](/home/op/experiments/weight-space-programming/scripts/pipeline.py:42) still records the hardcoded original revision. Authenticate the complete base—or all expected language-model tensors—before reconstruction.

4. **P2 — The documented workflow fails on a fresh clone.** [.gitignore:7](/home/op/experiments/weight-space-programming/.gitignore:7) excludes `upstream/`, but [pipeline.py:52](/home/op/experiments/weight-space-programming/scripts/pipeline.py:52) requires it before every action, including download. [README.md:15](/home/op/experiments/weight-space-programming/README.md:15) supplies no upstream bootstrap commands or commit pins. Document or automate fetching the exact required checkout.

5. **P2 — Successful reconstruction can fail when recording provenance.** [pipeline.py:43](/home/op/experiments/weight-space-programming/scripts/pipeline.py:43) writes into `reports/` without creating it. That directory is currently empty, so Git will not recreate it on clone. The command then fails after producing the checkpoint; retrying encounters the existing-output refusal. Create and validate the report destination before reconstruction.

6. **P2 — Failed writes leave an unusable output that blocks retries.** [st_numpy.py:46](/home/op/experiments/weight-space-programming/upstream/lm_understand-logs/exp8_math/code/st_numpy.py:46) writes directly to the final filename. Interruption or disk exhaustion leaves a partial checkpoint, which [pipeline.py:39](/home/op/experiments/weight-space-programming/scripts/pipeline.py:39) subsequently refuses to replace. Write to a temporary file, verify it, and publish atomically with overwrite protection.

7. **P2 — The wrapper has no regression tests.** The top-level `tests/` directory is empty. The paths at [pipeline.py:23](/home/op/experiments/weight-space-programming/scripts/pipeline.py:23) and [pipeline.py:35](/home/op/experiments/weight-space-programming/scripts/pipeline.py:35) need coverage for optimized execution, incorrect base/edit hashes, missing manifest entries, overwrite attempts, fresh-checkout setup, interrupted writes, and preservation of untouched tensors. OpenWSP’s separate tests do not validate this wrapper.

Review covered the wrapper and its invoked reconstruction code. No files were modified. Validation was static: shell startup failed, and the alternate process runner required unavailable approval, so no tests or rebuild were run.

# Direct-worker discrepancy repairs — 8 November 2026

All eight engineering discrepancies audited against
`f0f340213dd863d6d780bc52e35b20e4e17a2b08` are repaired by this change.
These are execution-integrity fixes. They do not repair Qwen weights or change
an experimental verdict. No Qwen checkpoint was loaded for this repair, and no
training, fitting, backward pass, optimizer, adapter merge, new corpus job,
model promotion or Hugging Face weight update was performed.

The original 14-forward study remains **`STOP_controls_failed`**. Reverse
transfer selected `>=` while retaining positive joint margin +0.8175669323;
the frozen criterion requires a negative margin as well. No subdivision,
coordinate selection, weight candidate or independent quality claim follows.
Paired model quality and new OpenCode/Hermes acceptance remain **`NOT_RUN`**.
See the [unchanged scientific account](../experiments/direct-weight-20261108.md)
and [paired acceptance boundary](../direct-paired-acceptance.md).

## Repairs and regression coverage

| ID | Discrepancy and consequence | Repair | Regression evidence |
| --- | --- | --- | --- |
| F01 | A locked parent could change after full SHA-256 verification but before reopening a tensor or loading the model. | Retain device/inode/size/mtime/ctime identities, pin read-only descriptors for headers, JSON and tensor reads, and check the authenticated generation before/after consumption and model loading. Recheck the baseline after inference. | Mutate a shard after real authentication; replace a sidecar inode with identical-size bytes; modify it during descriptor use; refuse changed inputs before importing model libraries; accept a stable parent. |
| F02 | Grading hashed mutable generation/request paths after executing their earlier contents. | Read, parse and hash each input from one byte snapshot before execution. Preserve exact input and grader-source copies. Worker and broker request bindings also use captured bytes. | Rewrite both source paths during mocked grading, and all worker inputs during mocked baseline authentication; results retain the consumed bytes' identities. |
| F03 | Concurrent brokers could both admit a 10 GiB worker from the same 14 GiB snapshot. | Hold one host/user `flock` from before resource sampling until confirmed worker exit. Persist container intent before launch, and refuse a live or unknown orphan. Never unlink the lease inode or infer exit from an old/reused broker PID. | Two actual broker processes with fake workers yield one launch and one refusal; test live/unknown orphan refusal, confirmed-exit recovery and inode preservation. |
| F04 | Observation interruption removed hooks but skipped the full parameter postcheck. | Put setup/generation in the preservation guard and hash all loaded parameters in `finally`. Persist an incomplete observation result and propagate it into the bound worker result. | Interrupt startup and generation; all 426 parameter hashes are checked and incomplete evidence is saved. |
| F05 | Completed mandatory contexts could conceal skipped requested optional controls behind `complete: true`. | Record `contrast_complete` separately. Overall completion and `all_requested_arms_completed` require every requested arm; host verification independently checks the exact ordered arm table. | Complete the contrast while refusing an optional arm on budget; overall completion remains false. |
| F06 | Donor loading trusted a cached diagnostic PASS with missing or failing raw contexts. | Require exactly one completed baseline, cue and length-control row, validate finite joint-score arithmetic, recompute the fixed gate and require exact agreement. Stage 1 ancestry also validates its full score table. Parse donor JSON/NPZ from their bound bytes. | Refuse both missing-baseline and failing-cue stale PASS records; retain valid donors, numeric archive and Stage 1/2 ancestry tests. |
| F07 | A complete worker report with one arbitrary parameter, zero arms and no gate could be accepted. | New evidence schema 2 binds frozen producer sources, full 12-file baseline proof, loaded-parent proof, exact 426-name before/after maps, tokenization, raw scores and numeric observations. Validate all declared arms and recompute diagnostic gates. | Positive synthetic worker proofs plus refusals for invented names, missing/duplicate/reordered/skipped arms, absent proofs, rewritten artifacts and forged verdicts. Completed execution with a failed scientific gate remains a non-quality result. |
| F08 | EOS at exactly the output cap was incorrectly marked censored. | Interpret the actual final token against model/tokenizer EOS identities. Record `finish_reason`, `reached_length_cap`, `hit_token_limit` and `censored` separately. Unknown short stops remain censored. | EOS at 192 and 384, either configured EOS ID, cap without EOS and short unknown stops; completed exact-cap output remains gradeable. |

Machine-readable tracking is in
[the findings record](20261108-direct-harness-fixes.json).
The original nine-test audit reproduction failed at all eight audited boundaries
before the fixes; its cap-without-EOS control already passed. The two donor
counterexamples are subtests. The expanded suite retains those reproductions
and adds reservation recovery, positive evidence and input-generation cases.
All tests use synthetic data or tiny random fixtures, never a Qwen quality trial.

## Validation and preservation

Run the full Linux CPU suite with the exact image identity used by CI:

```sh
WSP_TEST_DOCKER_IMAGE=python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54 \
  python -m unittest discover -s tests -v
```

The `.54` repair worktree is
`/home/op/work/weight-space-programming-harness-fixes-20261108`.
Its retained `reports/harness-fixes-20261108/` holds complete test logs, source
hashes, container state, resource observations and original file snapshots.
The initial Linux run's two failures were the test environment's shorter image
name (`python@sha256:…` rather than CI's `python:3.12-slim@sha256:…`), not a
changed digest. Its failed log is preserved. macOS is not the full Linux
validation target: `renameat2`, `/proc`, Docker and local socket access differ.
Final `.54` validation passed:

- **389 tests:** 375 passed, 14 skipped, zero failures/errors, 54.961 seconds.
  [Complete Linux transcript](20261108-direct-harness-validation/full-linux-02.log),
  SHA-256 `cce78ae3538a71e24adf3e80da3acae6f56ec949e0b9c8adfb3336b885a52acd`.
- **7/7 separate runtime tests passed**, zero skips/failures/errors, 11.908 seconds.
  [Runtime transcript](20261108-direct-harness-validation/runtime-tests.log),
  SHA-256 `b5dbb2cb4ccc7074815b4598142618ec446ba7864b47294fcb94a66c14dcb494`.
  [Frozen source identities](20261108-direct-harness-validation/runtime-source-sha256.json)
  match every corresponding file in this change.
- The 14 initial skips comprise those seven runtime tests plus seven historical
  training/PEFT/export/API checks. The latter remain intentionally unrun.
- **29 new regression test methods** cover the audited failures and related
  integrity edges. No new scientific/model evaluation was run.

Seven runtime checks use a separately admitted **2 GiB, one-CPU**, no-network,
read-only container with empty CUDA visibility: five exact-dtype/reference-kernel
tests and two tiny activation-hook/addition tests. They reuse existing inference
libraries read-only and mount no Qwen checkpoint. Legacy training API tests stay
skipped; no training environment is installed or repaired to run them.

No original experiment artifact or raw record is rewritten. Existing sources,
interrupted outputs and negative results remain available. The runtime checkout
`/home/op/work/weight-space-programming-20261106` and study worktree
`/home/op/work/weight-space-programming-direct-20261108` are not updated.
The stopped unit `wsp54-formatted-repair-20261108-01.service` remains stopped
(`MainPID=0`, `Restart=no`); its failed state is retained. This task issues no
service-control commands and does not change Spark Fable Runtime, Bonsai, Moss,
the separate evaluation or other jobs. Test containers are retained.

## Evidence compatibility and limits

`evidence_schema_version: 2` is separate from request schema 1 or 2. It is required
for new host acceptance. Historical results lacking it remain historical and
must be interpreted with their archived producer/source hashes. Do not edit
those results to make them look like new evidence or resume an old experiment
under changed source identities. Donor score checks can still inspect older
practice evidence, but fresh causal execution separately requires the frozen
runtime/source identity and replay gates. The closed hypothesis remains closed.

The lease serializes this user's direct brokers and admitted runtime tests. It
does not reserve memory against unrelated programs that ignore it; live reserve
monitoring and hard container limits remain necessary. On an unknown orphan,
inspect the retained exact container and report the blocker. Do not clear the
lease, stop other jobs or use PID age to bypass admission.

File-generation checks detect mutation after full authentication; they do not
turn an externally writable shared cache into immutable storage. A changed
parent must be fully authenticated again in a new run. The trusted repository,
lock, host kernel and producer code remain the authority boundary; these checks
are not a sandbox for an adversary replacing all trusted code. Hash consistency
and completed execution do not establish causal repair or independent quality.

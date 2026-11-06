# Benchmark interruption — 7 November 2026

The three GPUQ attempts were **cancelled**, not successfully completed. The original thinking, repaired thinking, and original non-thinking instances ran on physical GPUs 0, 1, and 2 respectively.

Model loading and engine initialization completed. All three entered their first `llm.generate` call. SIGINT shutdowns appeared at approximately **08:49:07, 08:49:13, and 08:49:18 NZDT**, respectively. The queue records have `cancel=1`, `status=cancelled`, and exit code 1.

**0/2,000 saved samples per arm; no benchmark accuracy or speed comparison is available.** Zero saved tokens is a lower bound, not proof of no generation: shutdown aborted in-flight requests before the batch was persisted. Those discarded-token totals cannot be recovered from the saved sample records.

## Attribution

The initiator of cancellation is **not recorded** in the evidence inspected. The bounded watchdog journal showed recent GPU activity and no stop event for these jobs. Do not label this an out-of-memory error, a watchdog timeout, or a voluntary user cancellation without additional evidence.

This audit did not cancel jobs, stop another workload, retry cancelled attempts, change admission thresholds, or modify frozen benchmark code/settings.

## Evidence and recovery boundary

Numeric queue metadata, selected engine lifecycle lines, source-log hashes and watchdog observations are in `benchmark-interruption-20261107.json`. Raw generation text and raw agent transcripts remain local.

The next approved run should preserve incremental per-sample completions and token counters so an interruption cannot erase the entire in-flight batch accounting. Such a change needs a new versioned experiment, not an edit to the current frozen manifest. No retry was submitted by this check.

# Actual token usage and benchmark status

Snapshot: **2026-11-07T08:54:29.252635+13:00 (Pacific/Auckland)**.

## Measured frontier-model usage

Recovered numeric usage receipts for three GPT-6.1 Sol Max sessions. These counts include repeated context processed across calls; they are not unique text length or an invoice.

| Stage | Input (includes cached) | Cached input subset | Output (includes reasoning) | Reasoning subset | Input + output |
|---|---:|---:|---:|---:|---:|
| Blocked implementation attempt | 1,057,241 | 991,872 | 7,168 | 4,415 | 1,064,409 |
| Static code review | 786,523 | 708,096 | 9,054 | 5,188 | 795,577 |
| Completed architecture plan | 33,691 | 12,288 | 10,358 | 7,250 | 44,049 |
| **Total recovered** | **1,877,455** | **1,712,256** | **26,580** | **16,853** | **1,904,035** |

Uncached input: **165,199**. Uncached input plus output: **191,779**. Cached tokens are not free tokens and are not excluded from the full total.

The older CLI totals of 72,537 (blocked implementation) and 87,481 (static review) equal input minus cached input plus output, not full token throughput. The structured receipt totals above include the cached input. The final plan used 44,049 input-plus-output tokens.

Each historical session was reconciled by deduplicating per-response receipts and matching their sum to the final cumulative counter. Cumulative snapshots were not added together. The plan completion event was matched to its previously committed provenance. Only numeric metadata and source hashes are saved; no private reasoning, prompts, raw agent transcripts, or benchmark outputs are uploaded.

### Outcomes and missing scope

- **Blocked implementation attempt**: Agent invocation completed but execution/write permissions blocked implementation. No code changes, tests, or commit were produced by this agent.
- **Static code review**: Completed a source-only review with seven findings; no tests or rebuild were run by this agent.
- **Completed architecture plan**: Plan-only Sol Max invocation completed. No tools, code changes or benchmark data access by the planner.

The deterministic CPU reconstruction does not call a frontier model. The surrounding ChatGPT authoring/review/tool-use tokens are not exposed here and remain **unknown**, not zero. No separate local Astra discovery invocation was recorded. The autonomous discovery process is not running; these totals do not establish its future token cost.

## RTX 3090 benchmark check

| Arm | Queue status | Physical GPU | Completed / expected | Saved prompt tokens | Saved generated tokens | Full accuracy |
|---|---|---:|---:|---:|---:|---:|
| original-thinking | cancelled | 0 | 0 / 2,000 | 0 | 0 | Pending |
| repaired-thinking | cancelled | 1 | 0 / 2,000 | 0 | 0 | Pending |
| original-nonthinking | cancelled | 2 | 0 / 2,000 | 0 | 0 | Pending |

Saved samples: **0/6,000**. Saved generation: **0 tokens**. These are local Qwen tokens, not Sol/Astra API tokens.

Saved sample totals are a lower bound on actual local computation while a job is running or after any failed attempt. Active batches, engine warmup and discarded/retried generation are not instrumented by this observer. Zero saved tokens does **not** prove zero GPU generation.

The protocol permits at most **196,608,000 generated tokens** across 6,000 samples. That is a configured ceiling, not consumed usage or a forecast.

All three arms retain the frozen BF16, MATH-500 x4, 32,768-output-token protocol. This observer does not start, stop, retry, or alter a GPU job, change the manifest, or make any model/API call.

## Refresh and evidence

Run from the project checkout with its existing environment:

```bash
.eval-venv/bin/python scripts/usage_audit.py \
  --manifest runs/math500-three-arm-20261107/manifest.json \
  --agents docs/usage/agent-token-usage-20261107.json \
  --json-output runs/math500-three-arm-20261107/token-usage-latest.json \
  --markdown-output runs/math500-three-arm-20261107/token-usage-latest.md
```

Imported agent evidence: `docs/usage/agent-token-usage-20261107.json`. Committed observation: `docs/usage/token-usage-snapshot-20261107.json`. Raw local source files remain outside Git. Refreshing the observer requires no extra Sol/Astra usage.

OpenAI accounting definitions: [Counting tokens](https://developers.openai.com/api/docs/guides/token-counting) and [Observability and usage](https://developers.openai.com/api/docs/guides/agents-api/observability). The numeric totals themselves come from the local execution receipts, not documentation examples.

## Interpretation limits

- The three recovered Sol sessions are not a complete accounting of this ChatGPT conversation or all project work.
- No actual autonomous weight-discovery run has been performed; these are implementation, review and planning usage.
- No OpenAI invoice or subscription credit ledger was queried; monetary cost is not measured.
- Cached input is included in input and reasoning output in output; do not add either subset again.
- Queue state is sampled sequentially; the report is a timestamped observation, not an atomic scheduler snapshot.
- Local totals count saved samples only. Unknown in-flight/warmup/discarded work is not assigned zero tokens.
- Partial sample accuracy is not a full MATH-500 result; quality comparisons require complete arms.

See also the [7 November interruption evidence](usage/benchmark-interruption-20261107.md): the original attempts entered generation but were cancelled before their first batch was persisted.

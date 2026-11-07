#!/usr/bin/env python3
"""Read-only token accounting for recorded agent sessions and saved GPU samples.

No provider calls, CUDA imports, queue mutations, or changes to frozen runs.
Only aggregate numeric metadata is exported; prompts, reasoning, tool arguments,
credentials and benchmark answer text are never copied into the report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
from zoneinfo import ZoneInfo

import compare_gpu as c

CORE = ('input_tokens', 'output_tokens')
BREAKDOWNS = ('cached_input_tokens', 'cache_write_input_tokens', 'reasoning_output_tokens')
RAW_FIELDS = CORE + BREAKDOWNS + ('total_tokens',)


def integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def normalize(raw: dict) -> dict:
    """Cached input and reasoning output are subsets, not extra usage."""
    result = {k: integer(raw[k], k) for k in CORE}
    for k in BREAKDOWNS:
        result[k] = integer(raw[k], k) if raw.get(k) is not None else None
    result['total_tokens'] = result['input_tokens'] + result['output_tokens']
    if 'total_tokens' in raw and integer(raw['total_tokens'], 'total_tokens') != result['total_tokens']:
        raise ValueError('Reported total does not equal input + output')
    for subset, parent in [('cached_input_tokens', 'input_tokens'), ('cache_write_input_tokens', 'input_tokens'), ('reasoning_output_tokens', 'output_tokens')]:
        if result[subset] is not None and result[subset] > result[parent]:
            raise ValueError(f'{subset} exceeds {parent}')
    cached = result['cached_input_tokens']
    result['noncached_input_tokens'] = None if cached is None else result['input_tokens'] - cached
    result['noncached_input_plus_output_tokens'] = None if cached is None else result['total_tokens'] - cached
    reasoning = result['reasoning_output_tokens']
    result['non_reasoning_output_tokens'] = None if reasoning is None else result['output_tokens'] - reasoning
    return result


def add_usage(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('No measured usage: unknown must not become zero')
    totals = {k: sum(r[k] for r in rows) for k in CORE}
    for k in BREAKDOWNS:
        totals[k] = None if any(r.get(k) is None for r in rows) else sum(r[k] for r in rows)
    return normalize(totals)


def rollout_usage(path: Path) -> dict:
    """Deduplicate response receipts and reconcile against the final cumulative counter.

    Deliberately do NOT sum cumulative token_count snapshots or both event formats.
    Mixed-model and forked/inherited sessions fail closed rather than overstate scope.
    """
    receipts, last, session, contexts, first, end = {}, None, None, set(), None, None
    with path.open() as source:
        for line in source:
            event = json.loads(line)
            payload = event.get('payload') or {}
            typ = event.get('type')
            if typ == 'session_meta':
                session = payload.get('id')
                first = payload.get('timestamp')
            elif typ == 'turn_context':
                contexts.add((payload.get('model'), payload.get('effort')))
            elif typ == 'token_usage_record':
                rid = payload.get('response_id')
                if not rid:
                    raise ValueError('Usage receipt lacks response identity')
                usage = normalize(payload['usage'])
                if rid in receipts and receipts[rid] != usage:
                    raise ValueError('Conflicting duplicate response usage')
                receipts[rid] = usage
            elif typ == 'event_msg' and payload.get('type') == 'token_count':
                info = payload.get('info') or {}
                if info.get('total_token_usage') is not None:
                    last = normalize(info['total_token_usage']); end = event.get('timestamp')
    if not session or len(contexts) != 1 or not receipts or last is None:
        raise ValueError('Incomplete or mixed-model session accounting')
    summed = add_usage(list(receipts.values()))
    if any(summed[k] != last[k] for k in RAW_FIELDS):
        raise ValueError('Per-response usage does not match final cumulative counter')
    model, effort = next(iter(contexts))
    return {'session_id': session, 'model': model, 'reasoning_effort': effort,
            'started_utc': first, 'last_usage_utc': end, 'usage': summed,
            'unique_response_receipts': len(receipts), 'reconciled_to_final_cumulative_counter': True,
            'source': {'kind': 'codex_rollout', 'filename': path.name, 'sha256': c.digest(path)}}


def plan_usage(events: Path, provenance: Path) -> dict:
    """Import the single completed turn of the existing ephemeral planner."""
    turns, thread = [], None
    with events.open() as source:
        for line in source:
            e = json.loads(line)
            if e.get('type') == 'thread.started': thread = e['thread_id']
            if e.get('type') == 'turn.completed': turns.append(normalize(e['usage']))
    p = json.loads(provenance.read_text())
    if len(turns) != 1 or not thread:
        raise ValueError('Expected one completed ephemeral planner turn')
    if [normalize(u) for u in p['usage']] != turns:
        raise ValueError('Planner event/provenance usage disagreement')
    return {'session_id': thread, 'model': p['model'], 'reasoning_effort': p['reasoning_effort'],
            'usage': turns[0], 'completed_turns': 1, 'unique_response_receipts': None,
            'reconciled_to_committed_provenance': True,
            'source': {'kind': 'codex_exec_events', 'filename': events.name, 'sha256': c.digest(events),
                       'provenance_sha256': c.digest(provenance)}}


def agent_totals(rows: list[dict]) -> dict:
    if len({r['session_id'] for r in rows}) != len(rows):
        raise ValueError('Same agent session imported more than once')
    return add_usage([normalize(r['usage']) for r in rows])


def saved_samples(m: dict, mh: str, arm: str, questions: list[dict]) -> dict:
    """Stream one atomic sample at a time; in-flight/failed computation is unmetered."""
    seen, inputs, outputs, correct, truncated, grading_errors = set(), 0, 0, 0, 0, 0
    directory = Path(m['run_dir']) / arm / 'records'
    for path in sorted(directory.glob('*.json.gz')):
        r = c.read_record(path)
        qi, si = r['question_index'], r['sample_index']
        if type(qi) is not int or type(si) is not int or not (0 <= qi < len(questions) and 0 <= si < m['samples_per_problem']):
            raise ValueError('Invalid sample index')
        if (qi, si) in seen:
            raise ValueError('Duplicate sample identity')
        if r.get('manifest_sha256') != mh or r.get('arm') != arm or r['question_id'] != questions[qi]['unique_id']:
            raise ValueError('Sample provenance mismatch')
        if r['seed'] != c.sample_seed(r['question_id'], si, m['seed']):
            raise ValueError('Sample seed mismatch')
        it = integer(r['prompt_tokens'], 'prompt_tokens')
        ot = integer(r['generated_tokens'], 'generated_tokens')
        if ot > m['sampling']['max_tokens']:
            raise ValueError('Sample exceeds frozen output cap')
        seen.add((qi, si)); inputs += it; outputs += ot
        correct += r['correct'] is True
        truncated += r['finish_reason'] == 'length'
        grading_errors += r['grading_error'] is not None
    expected = len(questions) * m['samples_per_problem']
    return {'completed_samples': len(seen), 'expected_samples': expected,
            'logical_prompt_tokens_saved_samples': inputs, 'generated_tokens_saved_samples': outputs,
            'total_tokens_saved_samples': inputs + outputs, 'correct_saved_samples': correct,
            'truncated_saved_samples': truncated, 'grading_errors_saved_samples': grading_errors,
            'full_arm_accuracy': correct / expected if len(seen) == expected else None,
            'coverage': 'Persisted successful sample records only; excludes in-flight batches, warmup, failed/retried/discarded work.'}


def read_queue(job_id: str, gpuq: str) -> dict:
    try:
        p = subprocess.run([gpuq, 'show', job_id], capture_output=True, text=True, timeout=15, check=True)
        value = json.loads(p.stdout)
        if value.get('id') != job_id: raise ValueError('Queue returned a different job')
        return {key: value.get(key) for key in ('id','name','status','assigned','gpu_count','reason','pid','exit_code','stop_reason','cancel','created','updated')}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {'id': job_id, 'status': 'unavailable', 'error': str(exc)[:400]}


def snapshot(manifest: Path, agents: Path, gpuq: str) -> dict:
    m, mh, questions = c.read_manifest(manifest)
    a = json.loads(agents.read_text())
    totals = agent_totals(a['sessions'])
    if totals != a['totals']: raise ValueError('Agent aggregate inconsistent with session receipts')
    receipt = json.loads((Path(m['run_dir']) / 'queue-receipt.json').read_text())
    if receipt['manifest_sha256'] != mh: raise ValueError('Queue receipt / manifest mismatch')
    arms = {}
    for arm in c.ARMS:
        folder = Path(m['run_dir']) / arm
        state = json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        if state and state.get('manifest_sha256') != mh: raise ValueError('Arm status manifest mismatch')
        summary = json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else None
        if summary and summary.get('manifest_sha256') != mh: raise ValueError('Arm summary manifest mismatch')
        failure = json.loads((folder/'failure.json').read_text()) if (folder/'failure.json').exists() else None
        arms[arm] = {'queue': read_queue(receipt['jobs'][arm]['job_id'], gpuq),
                     'last_reported_stage': state.get('status'), 'gpu_identity': state.get('identity'),
                     'samples': saved_samples(m, mh, arm, questions), 'summary_present': summary is not None,
                     'failure': failure}
    now = datetime.now(timezone.utc)
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.split(':',1)[0] in ('MemTotal','MemAvailable'):
            memory[line.split(':',1)[0]+'_bytes'] = int(line.split()[1])*1024
    return {'schema_version': 1, 'captured_utc': now.isoformat(),
            'captured_auckland': now.astimezone(ZoneInfo('Pacific/Auckland')).isoformat(),
            'manifest_sha256': mh, 'agent_usage_source_sha256': c.digest(agents),
            'frontier': a, 'benchmarks': arms,
            'local_tokens': {key: sum(v['samples'][key] for v in arms.values()) for key in
                             ('completed_samples','expected_samples','logical_prompt_tokens_saved_samples','generated_tokens_saved_samples','total_tokens_saved_samples')},
            'planned_output_token_upper_bound': m['problem_count'] * m['samples_per_problem'] * len(c.ARMS) * m['sampling']['max_tokens'],
            'host': {**memory, 'disk_free_bytes': shutil.disk_usage(m['run_dir']).free},
            'scope_limits': [
                'The three recovered Sol sessions are not a complete accounting of this ChatGPT conversation or all project work.',
                'No actual autonomous weight-discovery run has been performed; these are implementation, review and planning usage.',
                'No OpenAI invoice or subscription credit ledger was queried; monetary cost is not measured.',
                'Cached input is included in input and reasoning output in output; do not add either subset again.',
                'Queue state is sampled sequentially; the report is a timestamped observation, not an atomic scheduler snapshot.',
                'Local totals count saved samples only. Unknown in-flight/warmup/discarded work is not assigned zero tokens.',
                'Partial sample accuracy is not a full MATH-500 result; quality comparisons require complete arms.']}


def render(report: dict) -> str:
    a = report['frontier']; t = a['totals']
    lines = ['# Actual token usage and benchmark status', '',
             f"Snapshot: **{report['captured_auckland']} (Pacific/Auckland)**.", '',
             '## Measured frontier-model usage', '',
             'Recovered numeric usage receipts for three GPT-6.1 Sol Max sessions. These counts include repeated context processed across calls; they are not unique text length or an invoice.', '',
             '| Stage | Input (includes cached) | Cached input subset | Output (includes reasoning) | Reasoning subset | Input + output |',
             '|---|---:|---:|---:|---:|---:|']
    for s in a['sessions']:
        u = s['usage']
        lines.append(f"| {s['stage']} | {u['input_tokens']:,} | {u['cached_input_tokens']:,} | {u['output_tokens']:,} | {u['reasoning_output_tokens']:,} | {u['total_tokens']:,} |")
    lines += [f"| **Total recovered** | **{t['input_tokens']:,}** | **{t['cached_input_tokens']:,}** | **{t['output_tokens']:,}** | **{t['reasoning_output_tokens']:,}** | **{t['total_tokens']:,}** |", '',
              f"Uncached input: **{t['noncached_input_tokens']:,}**. Uncached input plus output: **{t['noncached_input_plus_output_tokens']:,}**. Cached tokens are not free tokens and are not excluded from the full total.", '',
              'The older CLI totals of 72,537 (blocked implementation) and 87,481 (static review) equal input minus cached input plus output, not full token throughput. The structured receipt totals above include the cached input. The final plan used 44,049 input-plus-output tokens.', '',
              'Each historical session was reconciled by deduplicating per-response receipts and matching their sum to the final cumulative counter. Cumulative snapshots were not added together. The plan completion event was matched to its previously committed provenance. Only numeric metadata and source hashes are saved; no private reasoning, prompts, raw agent transcripts, or benchmark outputs are uploaded.', '',
              '### Outcomes and missing scope', '']
    for s in a['sessions']: lines.append(f"- **{s['stage']}**: {s['outcome']}")
    lines += ['', 'The deterministic CPU reconstruction does not call a frontier model. The surrounding ChatGPT authoring/review/tool-use tokens are not exposed here and remain **unknown**, not zero. No separate local Astra discovery invocation was recorded. The autonomous discovery process is not running; these totals do not establish its future token cost.', '',
              '## RTX 3090 benchmark check', '',
              '| Arm | Queue status | Physical GPU | Completed / expected | Saved prompt tokens | Saved generated tokens | Full accuracy |',
              '|---|---|---:|---:|---:|---:|---:|']
    for arm, v in report['benchmarks'].items():
        s=v['samples']; identity=v.get('gpu_identity') or {}
        gpu=identity.get('physical_index')
        acc='Pending' if s['full_arm_accuracy'] is None else f"{s['full_arm_accuracy']:.2%}"
        lines.append(f"| {arm} | {v['queue']['status']} | {gpu if gpu is not None else 'Unassigned'} | {s['completed_samples']:,} / {s['expected_samples']:,} | {s['logical_prompt_tokens_saved_samples']:,} | {s['generated_tokens_saved_samples']:,} | {acc} |")
    local = report['local_tokens']
    lines += ['', f"Saved samples: **{local['completed_samples']:,}/{local['expected_samples']:,}**. Saved generation: **{local['generated_tokens_saved_samples']:,} tokens**. These are local Qwen tokens, not Sol/Astra API tokens.", '',
              'Saved sample totals are a lower bound on actual local computation while a job is running or after any failed attempt. Active batches, engine warmup and discarded/retried generation are not instrumented by this observer. Zero saved tokens does **not** prove zero GPU generation.', '',
              f"The protocol permits at most **{report['planned_output_token_upper_bound']:,} generated tokens** across 6,000 samples. That is a configured ceiling, not consumed usage or a forecast.", '',
              'All three arms retain the frozen BF16, MATH-500 x4, 32,768-output-token protocol. This observer does not start, stop, retry, or alter a GPU job, change the manifest, or make any model/API call.', '',
              '## Refresh and evidence', '',
              'Run from the project checkout with its existing environment:', '', '```bash',
              '.eval-venv/bin/python scripts/usage_audit.py \\',
              '  --manifest runs/math500-three-arm-20261107/manifest.json \\',
              '  --agents docs/usage/agent-token-usage-20261107.json \\',
              '  --json-output runs/math500-three-arm-20261107/token-usage-latest.json \\',
              '  --markdown-output runs/math500-three-arm-20261107/token-usage-latest.md', '```', '',
              'Imported agent evidence: `docs/usage/agent-token-usage-20261107.json`. Committed observation: `docs/usage/token-usage-snapshot-20261107.json`. Raw local source files remain outside Git. Refreshing the observer requires no extra Sol/Astra usage.', '',
              'OpenAI accounting definitions: [Counting tokens](https://developers.openai.com/api/docs/guides/token-counting) and [Observability and usage](https://developers.openai.com/api/docs/guides/agents-api/observability). The numeric totals themselves come from the local execution receipts, not documentation examples.', '',
              '## Interpretation limits', '']
    lines.extend('- ' + x for x in report['scope_limits'])
    return '\n'.join(lines) + '\n'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--agents', type=Path, required=True)
    p.add_argument('--gpuq', default='/home/op/.local/bin/gpuq')
    p.add_argument('--json-output', type=Path, required=True)
    p.add_argument('--markdown-output', type=Path, required=True)
    args = p.parse_args()
    # Observer outputs cannot replace input evidence or either frozen run input.
    forbidden = {args.manifest.resolve(), args.agents.resolve(), (args.manifest.parent/'queue-receipt.json').resolve()}
    if args.json_output.resolve() in forbidden or args.markdown_output.resolve() in forbidden or args.json_output.resolve() == args.markdown_output.resolve():
        p.error('Output path conflicts with evidence or the other output')
    report = snapshot(args.manifest, args.agents, args.gpuq)
    c.atomic_json(args.json_output, report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render(report))
    print(json.dumps({'captured_auckland': report['captured_auckland'],
                      'recovered_frontier_tokens': report['frontier']['totals']['total_tokens'],
                      'saved_local_generation_tokens': report['local_tokens']['generated_tokens_saved_samples'],
                      'saved_samples': report['local_tokens']['completed_samples'],
                      'states': {k: v['queue']['status'] for k,v in report['benchmarks'].items()},
                      'report': str(args.markdown_output)}, indent=2))


if __name__ == '__main__':
    main()

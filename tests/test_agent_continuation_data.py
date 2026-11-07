"""Grounded edit and once-only outcome controls for continuation data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from build_agent_flow_corpus import samples, conversations as previous_conversations
from build_agent_continuation_corpus import conversations
from coder_common import normalize_record
from prepare_coder_data import prepare, read_dataset


def fixture_controls(rows):
    return {row['key']: {'control_cwd': '/isolated/' + row['key'],
        'before': {'exit_code': 1, 'stdout': '', 'stderr': 'Ran 3 tests\nFAILED (failures=2, errors=1)\n'},
        'after': {'exit_code': 0, 'stdout': '', 'stderr': 'Ran 3 tests\nOK\n'}} for row in rows}


class AgentContinuationDataTests(unittest.TestCase):
    def test_exact_edits_match_read_content_and_replace_the_entire_invalid_function(self):
        rows = samples(1)
        for raw in conversations(rows, fixture_controls(rows)):
            record = normalize_record(raw)
            calls = [c for message in record['messages'] for c in message.get('tool_calls', [])]
            edits = [c for c in calls if c['function']['name'] in ('edit', 'patch')]
            if not edits: continue
            edit, = edits; arguments = edit['function']['arguments']
            path, old, new = ('filePath', 'oldString', 'newString') if edit['function']['name'] == 'edit' else ('path', 'old_string', 'new_string')
            self.assertEqual(arguments[path], calls[0]['function']['arguments'][path])
            observed = record['messages'][3]['content']
            lines = observed.splitlines()
            separator = ': ' if path == 'filePath' else '|'
            self.assertEqual('\n'.join(line.split(separator, 1)[1] for line in lines) + '\n', arguments[old])
            matching = [row for row in rows if row['initial'] == arguments[old]]
            self.assertEqual(len(matching), 1)
            self.assertEqual(arguments[new], matching[0]['fixed'])
            compile(arguments[new], 'authored-exact-edit.py', 'exec')

    def test_observation_only_trajectories_execute_once_and_copy_both_failure_counts(self):
        rows = samples(2); failures = successes = 0
        for raw in conversations(rows, fixture_controls(rows)):
            record = normalize_record(raw)
            if 'Report only JSON' not in record['messages'][1]['content']: continue
            calls = [c for m in record['messages'] for c in m.get('tool_calls', [])]
            self.assertEqual(len(calls), 1)
            self.assertIn(calls[0]['function']['name'], ('run_tests', 'bash', 'terminal'))
            actual = json.loads(record['messages'][-2]['content'])
            answer = json.loads(record['messages'][-1]['content'])
            self.assertEqual(answer, {key: actual[key] for key in ('returncode', 'fail_count')})
            self.assertEqual(answer['fail_count'], 3 if answer['returncode'] else 0)
            failures += bool(answer['returncode']); successes += not answer['returncode']
        self.assertEqual(failures, 32); self.assertEqual(successes, 32)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Atomic publication requires Linux')
    def test_previous_and_extended_clients_and_repair_states_keep_identical_family_splits(self):
        rows = samples(2); controls = fixture_controls(rows)
        previous = previous_conversations(rows, controls); extended = conversations(rows, controls)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); inputs = []; sources = []
            for i, records in enumerate((previous, extended)):
                path = root / (str(i) + '.jsonl'); inputs.append(path)
                path.write_text(''.join(json.dumps(row) + '\n' for row in records))
                sources.append({'name': 'authored-control-' + str(i), 'revision': 'fixture', 'license': 'owner-authored'})
            manifest = prepare(inputs, sources, root / 'prepared', seed=20261106)
            self.assertEqual(manifest['rejected'], {})
            _, splits = read_dataset(root / 'prepared')
            self.assertEqual(sum(map(len, splits.values())), len(previous) + len(extended))
            self.assertFalse({r['task_id'] for r in splits['train']} & {r['task_id'] for r in splits['validation']})
            for split in splits.values():
                for task in {row['task_id'] for row in split}:
                    self.assertEqual({row['source'] for row in split if row['task_id'] == task}, {s['name'] for s in sources})


if __name__ == '__main__': unittest.main()

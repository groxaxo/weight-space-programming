"""Audited boundary regressions. No real model, Docker or service is run."""
import ast
import copy
from contextlib import nullcontext
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import direct_weights as weights
import direct_forward as forward
import direct_probe as probe
import direct_causal as causal
from ops import direct_edit54 as host, grade_direct_practice as grade
import test_direct_weights as weight_fixtures
from test_direct_causal import bundle
from test_direct_probe import fixture as probe_fixture
from test_direct_grading import fixture as grading_fixture
from test_direct_forward import request as observation_request


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


class _Words:
    def __init__(self, width): self.value = np.full(width, 0x3f80, dtype='<u2')
    def detach(self): return self
    def contiguous(self): return self
    def view(self, _): return self
    def numpy(self): return self.value


class _Grid:
    def __init__(self, width): self.width = width
    def __getitem__(self, _): return _Words(self.width)


class _Down:
    def register_forward_hook(self, callback):
        self.callback = callback
        return SimpleNamespace(remove=lambda: None)


class _Model:
    def __init__(self, interrupt=False, eos=True):
        self.model = SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=_Down())) for _ in range(32)])
        self.interrupt, self.eos = interrupt, eos
        self.generation_config = SimpleNamespace(eos_token_id=[2, 4])
    def generate(self, **kwargs):
        for block in self.model.layers:
            block.mlp.down_proj.callback(block.mlp.down_proj, [_Grid(9216)], _Grid(2560))
        if self.interrupt: raise TimeoutError('synthetic inference interruption')
        return np.asarray([[1] + [3] * (kwargs['max_new_tokens'] - 1) + [4 if self.eos else 3]])
    def get_parameter(self, name): return name


class AuditedBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_changed_parent_after_authentication_is_refused(self):
        fixture = weight_fixtures.DirectWeightContract(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        altered = fixture.words.copy(); altered.flat[0] = 0x3f82
        edited = altered.copy(); edited.flat[7] = 0x3f81
        fixture.patch['tensors'][0].update(before_sha256=weights.sha_bytes(altered.tobytes()), after_sha256=weights.sha_bytes(edited.tobytes()))
        fixture.save(); original_verify = weights.verify_baseline
        def race(lock, base):
            verified = original_verify(lock, base)
            weight_fixtures.write_shard(fixture.base / fixture.shards[0], fixture.names[0], altered)
            return verified
        with mock.patch.object(weights, 'verify_baseline', side_effect=race), self.assertRaises(weights.ValidationError):
            weights.validate_patch(fixture.lock, fixture.base, fixture.patch_path)

    def test_grader_binds_bytes_consumed_before_execution(self):
        generation, request = grading_fixture()
        g, r = self.root / 'generation.json', self.root / 'request.json'
        g.write_text(json.dumps(generation)); r.write_text(json.dumps(request))
        gh, rh = digest(g), digest(r)
        def process(command, **kwargs):
            if command[:2] == ['docker', 'run']:
                changed = copy.deepcopy(generation); changed['text'] = 'raise SystemExit(0)'
                other = copy.deepcopy(request); other['cases'][0]['messages'][0]['content'] += ' changed'
                g.write_text(json.dumps(changed)); r.write_text(json.dumps(other))
                return subprocess.CompletedProcess(command, 73)
            if command[:2] == ['docker', 'inspect']:
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps([{'State': {'Status': 'exited', 'Error': '', 'ExitCode': 73}}]), stderr='')
            if command[:2] == ['docker', 'logs']: return subprocess.CompletedProcess(command, 0, stdout=b'', stderr=b'')
            raise AssertionError(command)
        target = self.root / 'grade'
        with mock.patch.object(sys, 'argv', ['grade', '--generation', str(g), '--request', str(r), '--output', str(target)]), mock.patch.object(grade.subprocess, 'run', side_effect=process), mock.patch('builtins.print'):
            grade.main()
        report = weights.read_json(target / 'result.json')
        self.assertEqual(report['generation_sha256'], gh)
        self.assertEqual(report['request_sha256'], rh)
        self.assertNotEqual(report['generation_sha256'], digest(g))
        self.assertEqual((target / 'task/solution.py').read_text(), generation['text'])

    def observe(self, *, interrupt=False, eos=True, startup=False):
        request = observation_request(); request['cases'][0]['id'] = 'half_open_overlap-r1'
        generation, case = grading_fixture(); request['cases'][0]['messages'] = case['cases'][0]['messages']
        model = _Model(interrupt, eos)
        tokenizer = SimpleNamespace(eos_token_id=2, apply_chat_template=lambda *args, **kwargs: np.array([[1]]), decode=lambda *args, **kwargs: generation['text'])
        torch = ModuleType('torch'); torch.__version__ = 'synthetic'; torch.Tensor = np.ndarray
        torch.uint16 = 'uint16'; torch.ones_like = np.ones_like
        def seed(_):
            if startup: raise TimeoutError('synthetic startup interruption')
        torch.manual_seed = seed
        torch.inference_mode = nullcontext; torch.is_grad_enabled = lambda: False
        before = {f'parameter-{i}': '0' * 64 for i in range(426)}
        with mock.patch.dict(sys.modules, {'torch': torch}), mock.patch.object(forward, 'load_text_model', return_value=(model, tokenizer, {'official_dtype_restorations': [], 'cpu_reference_kernels': {'kind': 'synthetic'}}, before)), mock.patch.object(forward, 'parameter_sha256', return_value='0' * 64) as hashes, mock.patch('builtins.print'):
            if interrupt or startup:
                with self.assertRaises(TimeoutError): forward.observe(request, weights.load_lock(), {}, self.root)
                return hashes.call_count
            return forward.observe(request, weights.load_lock(), {}, self.root), request

    def test_interrupted_observation_rechecks_loaded_parameters(self):
        self.assertEqual(self.observe(interrupt=True), 426)
        report = weights.read_json(self.root / 'observation-result.json')
        self.assertFalse(report['complete']); self.assertTrue(report['parameter_postcheck_complete'])
        self.assertTrue(report['parameters_unchanged'])

    def test_observer_startup_interruption_also_rechecks_all_parameters(self):
        self.assertEqual(self.observe(startup=True), 426)
        report = weights.read_json(self.root / 'observation-result.json')
        self.assertFalse(report['complete']); self.assertTrue(report['parameter_postcheck_complete'])
        self.assertIn('startup interruption', report['error'])

    def test_eos_at_exact_cap_is_not_censored(self):
        report, request = self.observe(); row = report['results'][0]
        self.assertEqual(row['generated_tokens'], request['generation']['max_new_tokens'])
        self.assertEqual(row['output_ids'][-1], 4)
        self.assertFalse(row['hit_token_limit']); self.assertEqual(row['finish_reason'], 'eos')
        grade.prepare(row, request)

    def test_cap_without_eos_stays_censored(self):
        report, request = self.observe(eos=False)
        self.assertTrue(report['results'][0]['hit_token_limit'])
        with self.assertRaises(weights.ValidationError): grade.prepare(report['results'][0], request)

    def test_stale_donor_pass_refused_for_missing_or_failing_contexts(self):
        for missing in (False, True):
            root = self.root / str(missing); root.mkdir()
            tok, request, patient, source, lock, result, arrays, runtime = bundle(root)
            if missing: result['arms'] = [row for row in result['arms'] if row['id'] != 'baseline']
            else:
                cue = next(row for row in result['arms'] if row['id'] == 'cue')
                cue['next_token_id'] = cue['branches']['incorrect']['ids'][0]; cue['correct_minus_incorrect_logprob'] = -8
                size = len(cue['branches']['correct']['ids'])
                cue['branches']['correct'].update(token_logprobs=[-18 / size] * size, sequence_logprob=-18)
            self.assertNotEqual(probe.first_assertion_gate(result['arms'])['status'], 'diagnostic_gate_passed')
            (root / 'result.json').write_text(json.dumps(result)); request['donor_files']['result.json'] = digest(root / 'result.json')
            with self.subTest(missing=missing), self.assertRaises(weights.ValidationError): causal.load_donors(request, lock, root, patient, source)

    def test_skipped_requested_control_makes_probe_incomplete(self):
        tokenizer, request, patient, source, lock = probe_fixture()
        request['arms'] = [{'id': 'declared-control', 'source': 'cue', 'group': 0, 'strength': 0.5}]
        request['budget'].update(controls_admitted=True, max_forward_calls=3, max_wall_seconds=1700)
        model = SimpleNamespace(get_parameter=lambda name: name); before = {'protected.parameter': 'd' * 64}
        def execute(_model, _tok, _plan, _outdir, name, **kwargs): return {'id': name, 'status': 'completed', 'seconds': 0.01}, {}
        with mock.patch.object(probe, 'load_text_model', return_value=(model, tokenizer, {}, before)), mock.patch.object(probe, 'prepare_plans', return_value={'baseline': {}, 'cue': {}}), mock.patch.object(probe, 'run_forward', side_effect=execute), mock.patch.object(probe, 'parameter_sha256', return_value='d' * 64), mock.patch.object(probe, 'forward_admitted', side_effect=[True, True, False]):
            report = probe.run_probe(request, lock, {}, self.root, patient, source, request_sha256='f' * 64, patient_sha256='b' * 64, source_request_sha256='c' * 64)
        self.assertFalse(report['complete']); self.assertFalse(report['all_requested_arms_completed'])
        self.assertTrue(report['contrast_complete'])

    def test_single_parameter_no_arms_is_refused_as_complete_evidence(self):
        from test_direct_causal_host import request
        value = request(); value['lock_sha256'] = weights.load_lock()['_sha256']; (self.root / 'causal').mkdir()
        report = {'request_sha256': 'f' * 64, 'protocol_sha256': value['protocol_sha256'], 'parent': {'model_id': weights.MODEL_ID, 'revision': weights.REVISION, 'lock_sha256': value['lock_sha256']}, 'training': False, 'role': 'practice', 'status': 'practice_only', 'weights_edited': False, 'quality_evaluation': 'NOT_RUN', 'patient_sha256': value['patient_sha256'], 'source_request_sha256': value['source_request_sha256'], 'stage': 1, 'question': value['question'], 'donor_files': value['donor_files'], 'parameters_unchanged': True, 'parameter_postcheck_complete': True, 'complete': True, 'loaded_parameter_sha256_before': {'bogus': '0' * 64}, 'loaded_parameter_sha256_after': {'bogus': '0' * 64}}
        (self.root / 'causal/result.json').write_text(json.dumps(report))
        with self.assertRaises(weights.ValidationError): host.worker_evidence(self.root, 'causal', value, 'f' * 64, value['lock_sha256'])


def _broker_fixture(root_string, suffix, messages, release):
    root = Path(root_string); request = observation_request()
    path = root / ('request-' + suffix + '.json'); path.write_text(json.dumps(request))
    def freeze(destination):
        (destination / 'configs').mkdir(parents=True)
        (destination / 'configs/qwen35-4b-direct.lock.json').write_bytes((ROOT / 'configs/qwen35-4b-direct.lock.json').read_bytes())
        return {}
    class Child:
        returncode = 0
        def __init__(self, *args, **kwargs): messages.put('launched')
        def poll(self): return 0 if release.is_set() else None
        def wait(self, timeout=None): release.wait(timeout); return 0
    def process(command, **kwargs):
        if command[:2] == ['docker', 'inspect']:
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps([{'State': {'Status': 'exited', 'Running': False, 'Restarting': False, 'ExitCode': 0}}]), stderr='')
        return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
    try:
        with mock.patch.object(sys, 'argv', ['broker', 'observe', '--baseline', str(root / 'model'), '--deps', str(root / 'deps'), '--request', str(path), '--output', str(root / ('run-' + suffix))]), mock.patch.object(host.os, 'uname', return_value=SimpleNamespace(nodename='4790')), mock.patch.object(host, 'resources', return_value={'mem_available_bytes': 14 * host.GIB, 'disk_free_bytes': 3 * host.GIB}), mock.patch.object(host, 'freeze_source', side_effect=freeze), mock.patch.object(host, 'mounted_files', return_value=[]), mock.patch.object(host, 'DEFAULT_LEASE_PATH', root / 'shared-worker.lock', create=True), mock.patch.object(host.subprocess, 'Popen', Child), mock.patch.object(host.subprocess, 'run', side_effect=process), mock.patch.object(host.signal, 'signal'), mock.patch.object(host.time, 'sleep', return_value=None), mock.patch.object(host, 'worker_evidence', return_value={'worker_complete': True}):
            host.main()
    except weights.ValidationError: messages.put('refused')
    except BaseException as error: messages.put('ERROR:' + repr(error))


@unittest.skipUnless('fork' in multiprocessing.get_all_start_methods(), 'POSIX concurrency test')
class BrokerAdmissionRegression(unittest.TestCase):
    def test_concurrent_brokers_cannot_both_launch_on_same_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'model').mkdir(); (root / 'deps').mkdir()
            context = multiprocessing.get_context('fork'); messages = context.Queue(); release = context.Event()
            children = [context.Process(target=_broker_fixture, args=(directory, str(i), messages, release)) for i in range(2)]
            for child in children: child.start()
            try:
                self.assertEqual(sorted(messages.get(timeout=5) for _ in children), ['launched', 'refused'])
            finally:
                release.set()
                for child in children:
                    child.join(timeout=5)
                    if child.is_alive(): child.terminate(); child.join(timeout=5)
                messages.close()


if __name__ == '__main__': unittest.main()

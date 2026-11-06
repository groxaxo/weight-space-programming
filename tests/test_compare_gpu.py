import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import compare_gpu as c
import gpu_admission as a


class EvaluationTests(unittest.TestCase):
    def test_nested_boxes(self):
        self.assertEqual(c.extract_answer(r'answer \boxed{\frac{1}{2}}'), r'\frac{1}{2}')

    def test_unfinished_last_box_not_prior_answer(self):
        self.assertIsNone(c.extract_answer(r'\boxed{5} then \boxed{7'))

    def test_last_reasoning_block_removed(self):
        self.assertEqual(c.extract_answer(r'<think>\boxed{1}</think>answer \boxed{2}'), '2')
        self.assertIsNone(c.extract_answer(r'<think>\boxed{1}</think>no final answer'))

    def test_fbox_and_bare_box(self):
        self.assertEqual(c.extract_answer(r'\fbox{7}'), '7')
        self.assertEqual(c.extract_answer('\\boxed 8$ trailing'), '8')
        self.assertIsNone(c.extract_answer(r'\boxed{}'))

    def test_published_box_precedence(self):
        self.assertEqual(c.extract_answer(r'\boxed{2} then \fbox{3}'), '2')

    def test_seed_is_matched_across_arms_and_distinct_across_samples(self):
        self.assertEqual(c.sample_seed('q1', 0, 4), c.sample_seed('q1', 0, 4))
        self.assertEqual(len({c.sample_seed('q1', s, 4) for s in range(4)}), 4)

    def test_gpu_not_initialized_on_module_import(self):
        self.assertNotIn('vllm', sys.modules)
        self.assertNotIn('torch', sys.modules)

    def test_queue_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError): c.queue_identity()

    def test_multi_gpu_rejected(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'GPU-a,GPU-b','GPU_QUEUE_JOB_ID':'x','GPU_QUEUE_PHYSICAL_GPUS':'0,1'}, clear=True):
            with self.assertRaises(RuntimeError): c.queue_identity()

    def test_one_gpu_accepted_without_remapping(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'GPU-a','GPU_QUEUE_JOB_ID':'x','GPU_QUEUE_PHYSICAL_GPUS':'2'}, clear=True):
            self.assertEqual(c.queue_identity()['physical_index'], 2)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-a')

    def test_record_atomic_no_clobber(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'sample.json.gz'
            c.save_record(p, {'x': 1})
            with self.assertRaises(FileExistsError): c.save_record(p, {'x': 2})
            self.assertEqual(c.read_record(p), {'x': 1})
            self.assertEqual([f.name for f in Path(d).iterdir()], ['sample.json.gz'])

    def test_atomic_json_deterministic_hash(self):
        self.assertEqual(c.canonical_hash({'a':1, 'b':2}), c.canonical_hash({'b':2, 'a':1}))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a'/'result.json'; c.atomic_json(p, {'good': True})
            self.assertEqual(json.loads(p.read_text()), {'good': True})

    def test_question_bootstrap_constant(self):
        self.assertEqual(c.bootstrap_interval([.5]*10), [.5, .5])

    def test_admission_resource_unique_per_job(self):
        self.assertNotEqual(a.resource({'id':'one'}), a.resource({'id':'two'}))

    def test_proc_identity_reads_current_process(self):
        self.assertIsNotNone(a.starttime(os.getpid()))
        self.assertIsNone(a.starttime(999999999))

    def test_admission_low_disk_is_wait_not_kill(self):
        with tempfile.TemporaryDirectory() as d, patch.object(a,'STATE',Path(d)), patch.object(a.shutil,'disk_usage',return_value=type('D',(),{'free':1})()):
            self.assertIn('free disk', a.check({'id':'test'}))

    def test_admission_low_ram_is_wait(self):
        with tempfile.TemporaryDirectory() as d, patch.object(a,'STATE',Path(d)), patch.object(a.shutil,'disk_usage',return_value=type('D',(),{'free':100*a.GIB})()), patch.object(a,'available_memory',return_value=1):
            self.assertIn('RAM', a.check({'id':'test'}))

    def test_admission_acquire_contract_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d, patch.object(a,'STATE',Path(d)), patch.object(a,'reason',return_value=None):
            with a.acquire({'id':'test'}) as lease:
                self.assertEqual(lease, {'env':{},'fds':()})
                self.assertTrue((Path(d)/'test.json').exists())
            self.assertFalse((Path(d)/'test.json').exists())

    def test_completed_records_require_full_identity(self):
        with tempfile.TemporaryDirectory() as d:
            m={'run_dir':d,'seed':0}; row={'unique_id':'q1'}
            p=Path(d)/'original-thinking'/'records'/'0000-0.json.gz'
            r={'manifest_sha256':'expected','arm':'original-thinking','question_index':0,'sample_index':0,'question_id':'q1','seed':c.sample_seed('q1',0,0)}
            c.save_record(p,r)
            self.assertEqual(len(c.load_records(m,'expected','original-thinking',[row])),1)
            with self.assertRaises(ValueError): c.load_records(m,'WRONG','original-thinking',[row])


if __name__ == '__main__': unittest.main()

"""CPU-only regression tests. Fixtures are tiny synthetic BF16 checkpoints."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pipeline as p


def spec(path):
    return {"size": path.stat().st_size, "sha256": p.sha256(path)}


def write_tensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, values in tensors.items():
        blob = values.astype("<u2").tobytes()
        header[name] = {"dtype": "BF16", "shape": list(values.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    header["__metadata__"] = {"format": "pt"}
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "models/base"
        self.assets = self.root / "assets/p50c"
        self.out = self.root / "models/rebuilt"
        self.base.mkdir(parents=True)
        self.assets.mkdir(parents=True)
        self.shard = "model.safetensors-00001-of-00001.safetensors"
        self.names = [f"model.language_model.layers.{n}.mlp.down_proj.weight" for n in (15, 19)]
        self.tensors = {name: (np.arange(6, dtype=np.uint16) + 100 * i).reshape(2, 3)
                        for i, name in enumerate(self.names, 1)}
        self.tensors["untouched.weight"] = np.array([777, 778, 779], dtype=np.uint16)
        write_tensors(self.base / self.shard, self.tensors)
        (self.base / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {n: self.shard for n in self.tensors}, "metadata": {"total_size": 30}}))
        (self.base / "config.json").write_text('{"model_type":"synthetic-test"}')
        (self.base / "tokenizer.json").write_text('{"version":"1.0"}')
        self.manifest = {"tensors": [], "changed_tensors": 2, "changed_entries_total": 4}
        self.arrays = {}
        for i, name in enumerate(self.names):
            original = self.tensors[name]
            idx = np.array([0, 3], dtype=np.int32)
            val = np.array([999, 1000], dtype=np.uint16)
            edited = original.copy().reshape(-1)
            edited[idx] = val
            self.manifest["tensors"].append({
                "name": name, "shape": [2, 3], "dtype": "bfloat16", "form": "sparse",
                "changed_entries": 2, "base_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
                "edited_sha256": hashlib.sha256(edited.tobytes()).hexdigest()})
            self.arrays[f"t{i}_idx"] = idx
            self.arrays[f"t{i}_val"] = val
        (self.assets / "manifest.json").write_text(json.dumps(self.manifest))
        np.savez(self.assets / "edit.npz", **self.arrays)
        self.lock = {
            "schema_version": 1,
            "source": {"repo": "example/research", "revision": "a" * 40,
                       "directory": "edits/p50c", "files": {n: spec(self.assets / n) for n in ("manifest.json", "edit.npz")}},
            "model": {"repo": "example/model", "revision": "b" * 40,
                      "files": {f.name: spec(f) for f in self.base.iterdir()}},
            "expected_tensors": self.names}
        (self.root / "scripts").mkdir()
        shutil.copyfile(Path(p.__file__), self.root / "scripts/pipeline.py")
        self.save_lock()

    def save_lock(self):
        (self.root / "reproduction.lock.json").write_text(json.dumps(self.lock))

    def update_manifest(self):
        (self.assets / "manifest.json").write_text(json.dumps(self.manifest))
        self.lock["source"]["files"]["manifest.json"] = spec(self.assets / "manifest.json")
        self.save_lock()

    def update_archive(self):
        np.savez(self.assets / "edit.npz", **self.arrays)
        self.lock["source"]["files"]["edit.npz"] = spec(self.assets / "edit.npz")
        self.save_lock()

    def run_cli(self, action, *extra, flags=(), env=None):
        return subprocess.run([sys.executable, *flags, str(self.root / "scripts/pipeline.py"), action, *extra],
                              capture_output=True, text=True, timeout=60, env=env)

    def build(self):
        return p.rebuild(self.lock, self.base, self.assets, self.out)

    def assert_no_partial_output(self):
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.out.parent.glob(".rebuilt.building-*")), [])

    def test_index_resolves_actual_hf_shard_filename(self):
        self.assertFalse((self.base / "model.safetensors").exists())
        result = p.verify(self.lock, self.base, self.assets)
        self.assertEqual(result["changed_entries_total"], 4)
        self.assertEqual(len(result["tensors"]), 2)

    def test_explicit_shard_path_remains_supported(self):
        self.assertEqual(p.verify(self.lock, self.base / self.shard, self.assets)["status"], "verified")

    def test_rebuild_copies_metadata_and_preserves_base(self):
        before = {f.name: p.sha256(f) for f in self.base.iterdir()}
        report = self.build()
        self.assertEqual(report["changed_entries_total"], 4)
        for name in self.lock["model"]["files"]:
            self.assertTrue((self.out / name).is_file())
            self.assertEqual(p.sha256(self.base / name), before[name])
        self.assertFalse((self.root / "reports").exists())
        self.assertEqual(json.loads((self.out / "rebuild.json").read_text()), report)
        self.assertTrue(p.verify_output(self.lock, self.base, self.assets, self.out)["all_other_bytes_preserved"])

    def test_output_requires_intact_reconstruction_provenance(self):
        self.build()
        path = self.out / "rebuild.json"
        original = path.read_text()
        for key, value in (("base_revision", "f" * 40), ("changed_entries_total", 999),
                           ("lock_sha256", "0" * 64), ("output_file_sha256", {})):
            report = json.loads(original)
            report[key] = value
            path.write_text(json.dumps(report))
            with self.subTest(key=key), self.assertRaises(p.ValidationError):
                p.verify_output(self.lock, self.base, self.assets, self.out)
        path.unlink()
        with self.assertRaises(p.ValidationError):
            p.verify_output(self.lock, self.base, self.assets, self.out)

    def test_corruption_in_unedited_base_is_rejected(self):
        path = self.base / self.shard
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        with self.assertRaisesRegex(p.ValidationError, "SHA-256"):
            self.build()
        self.assert_no_partial_output()

    def test_modified_config_is_rejected(self):
        path = self.base / "config.json"
        data = bytearray(path.read_bytes()); data[-2] ^= 1; path.write_bytes(data)
        with self.assertRaises(p.ValidationError):
            self.build()

    def test_truncated_base_is_rejected(self):
        path = self.base / self.shard
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(p.ValidationError, "Size mismatch"):
            p.verify(self.lock, self.base, self.assets)

    def test_modified_manifest_is_rejected_without_relocking(self):
        self.manifest["tensors"].pop()
        (self.assets / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaises(p.ValidationError):
            self.build()

    def test_manifest_cannot_omit_expected_tensor(self):
        self.manifest["tensors"].pop()
        self.update_manifest()
        with self.assertRaisesRegex(p.ValidationError, "Unexpected, missing"):
            self.build()

    def test_manifest_cannot_reorder_tensors(self):
        self.manifest["tensors"].reverse(); self.update_manifest()
        with self.assertRaisesRegex(p.ValidationError, "reordered"):
            self.build()

    def test_archive_integrity_enforced(self):
        path = self.assets / "edit.npz"
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        with self.assertRaisesRegex(p.ValidationError, "SHA-256"):
            self.build()

    def test_tensor_hash_checked_even_when_whole_base_lock_updated(self):
        path = self.base / self.shard
        header, offset = p.read_header(path)
        data = bytearray(path.read_bytes()); data[offset] ^= 1; path.write_bytes(data)
        self.lock["model"]["files"][self.shard] = spec(path)
        with self.assertRaisesRegex(p.ValidationError, "Base tensor mismatch"):
            self.build()
        self.assert_no_partial_output()

    def test_wrong_edited_hash_rejected(self):
        self.manifest["tensors"][0]["edited_sha256"] = "0" * 64
        self.update_manifest()
        with self.assertRaisesRegex(p.ValidationError, "Edited tensor mismatch"):
            self.build()
        self.assert_no_partial_output()

    def test_wrong_tensor_shape_rejected(self):
        self.manifest["tensors"][0]["shape"] = [1, 6]; self.update_manifest()
        with self.assertRaisesRegex(p.ValidationError, "dtype/shape"):
            self.build()

    def test_lowrank_edit_refused_explicitly(self):
        self.manifest["tensors"][0]["form"] = "lowrank"; self.update_manifest()
        with self.assertRaisesRegex(p.ValidationError, "Only this published sparse"):
            self.build()

    def test_invalid_sparse_indices_rejected(self):
        for indices in ([-1, 3], [0, 6], [0, 0]):
            with self.subTest(indices=indices):
                self.arrays["t0_idx"] = np.array(indices, dtype=np.int32)
                self.update_archive()
                with self.assertRaises(p.ValidationError):
                    self.build()
                self.assert_no_partial_output()

    def test_pickle_objects_are_not_loaded(self):
        self.arrays["t0_idx"] = np.array([{}, {}], dtype=object)
        self.update_archive()
        with self.assertRaisesRegex(ValueError, "Object arrays"):
            self.build()

    def test_extra_archive_array_rejected(self):
        self.arrays["unexpected"] = np.array([1]); self.update_archive()
        with self.assertRaisesRegex(p.ValidationError, "Unexpected edit archive"):
            self.build()

    def test_existing_output_directory_is_not_overwritten(self):
        self.out.mkdir(); (self.out / "keep").write_text("mine")
        with self.assertRaisesRegex(p.ValidationError, "existing output"):
            self.build()
        self.assertEqual((self.out / "keep").read_text(), "mine")

    def test_existing_output_file_is_not_overwritten(self):
        self.out.write_text("mine")
        with self.assertRaises(p.ValidationError):
            self.build()
        self.assertEqual(self.out.read_text(), "mine")

    def test_dangling_output_symlink_is_not_overwritten(self):
        self.out.symlink_to(self.root / "missing")
        with self.assertRaises(p.ValidationError):
            self.build()
        self.assertTrue(self.out.is_symlink())

    def test_nested_output_cannot_modify_base(self):
        with self.assertRaisesRegex(p.ValidationError, "overlap"):
            p.rebuild(self.lock, self.base, self.assets, self.base / "new")

    def test_legacy_file_output_has_actionable_error(self):
        with self.assertRaisesRegex(p.ValidationError, "DIRECTORY"):
            p.rebuild(self.lock, self.base, self.assets, self.root / "out.safetensors")

    def test_copy_failure_cleans_own_staging_only(self):
        stale = self.out.parent / ".rebuilt.building-other"; stale.mkdir()
        original = p.shutil.copyfile
        def interrupted(source, target):
            original(source, target)
            raise OSError("simulated disk failure")
        with mock.patch.object(p.shutil, "copyfile", side_effect=interrupted):
            with self.assertRaises(OSError): self.build()
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.out.parent.glob(".rebuilt.building-*")), [stale])

    def test_report_failure_does_not_publish(self):
        original = Path.write_text
        def fail_report(path, *args, **kwargs):
            if path.name == "rebuild.json": raise OSError("report write failed")
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, "write_text", fail_report):
            with self.assertRaises(OSError): self.build()
        self.assert_no_partial_output()

    def test_publication_race_never_replaces_empty_directory(self):
        original = p.publish_directory
        def competing_writer(stage, output):
            output.mkdir()
            original(stage, output)
        with mock.patch.object(p, "publish_directory", side_effect=competing_writer):
            with self.assertRaises(FileExistsError): self.build()
        self.assertTrue(self.out.is_dir())
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertEqual(list(self.out.parent.glob(".rebuilt.building-*")), [])

    def test_concurrent_rebuilds_publish_exactly_once(self):
        cmd = [sys.executable, str(self.root / "scripts/pipeline.py"), "rebuild"]
        first = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        second = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logs = [first.communicate(timeout=60), second.communicate(timeout=60)]
        self.assertEqual(sorted([first.returncode, second.returncode]), [0, 1], logs)
        self.assertEqual(list(self.out.parent.glob(".rebuilt.building-*")), [])
        p.verify_output(self.lock, self.base, self.assets, self.out)

    def test_optimized_modes_enforce_overwrite_protection(self):
        self.build()
        before = p.sha256(self.out / self.shard)
        modes = [(('-O',), os.environ.copy()), (('-OO',), os.environ.copy()),
                 ((), dict(os.environ, PYTHONOPTIMIZE="1"))]
        for flags, env in modes:
            with self.subTest(flags=flags, env_opt=env.get("PYTHONOPTIMIZE")):
                result = self.run_cli("rebuild", flags=flags, env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Refusing existing output", result.stderr)
                self.assertEqual(p.sha256(self.out / self.shard), before)

    def test_optimized_mode_enforces_tensor_hash(self):
        self.manifest["tensors"][0]["edited_sha256"] = "0" * 64; self.update_manifest()
        result = self.run_cli("verify", flags=("-O",), env=dict(os.environ, PYTHONOPTIMIZE="2"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Edited tensor mismatch", result.stderr)

    def test_stage_is_rechecked_after_source_copy(self):
        original = p.shutil.copyfile
        def corrupt_copy(source, target):
            result = original(source, target)
            if target.name == self.shard:
                data = bytearray(target.read_bytes()); data[-1] ^= 1; target.write_bytes(data)
            return result
        with mock.patch.object(p.shutil, "copyfile", side_effect=corrupt_copy):
            with self.assertRaisesRegex(p.ValidationError, "SHA-256"):
                self.build()
        self.assert_no_partial_output()

    def test_output_corruption_is_detected_inside_and_outside_edits(self):
        self.build()
        target = self.out / self.shard
        original = target.read_bytes()
        _, offset = p.read_header(target)
        for position in (offset, len(original) - 1):
            with self.subTest(position=position):
                data = bytearray(original); data[position] ^= 1; target.write_bytes(data)
                with self.assertRaises(p.ValidationError):
                    p.verify_output(self.lock, self.base, self.assets, self.out)
        target.write_bytes(original)

    def test_low_disk_fails_without_partial_output(self):
        usage = p.shutil.disk_usage(self.root)._replace(free=0)
        with mock.patch.object(p.shutil, "disk_usage", return_value=usage):
            with self.assertRaisesRegex(p.ValidationError, "bytes free"):
                self.build()
        self.assert_no_partial_output()

    def test_missing_assets_does_not_gate_download(self):
        contents = {f.name: f.read_bytes() for f in self.base.iterdir()}
        fresh = self.root / "downloaded"
        shutil.rmtree(self.assets)
        with mock.patch.object(p, "urlopen", side_effect=lambda u, **k: io.BytesIO(contents[u.rsplit('/', 1)[1]])):
            p.download(self.lock, fresh)
        self.assertEqual(p.sha256(fresh / self.shard), self.lock["model"]["files"][self.shard]["sha256"])

    def test_fresh_bootstrap_is_pinned_and_idempotent(self):
        data = {f.name: f.read_bytes() for f in self.assets.iterdir()}
        shutil.rmtree(self.assets)
        calls = []
        def response(url, **kwargs):
            calls.append(url)
            return io.BytesIO(data[url.rsplit('/', 1)[1]])
        with mock.patch.object(p, "urlopen", side_effect=response):
            p.bootstrap(self.lock, self.assets)
            p.bootstrap(self.lock, self.assets)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all("/" + "a" * 40 + "/" in url for url in calls))

    def test_bad_download_never_publishes_partial_file(self):
        destination = self.root / "downloads/asset"
        expected = {"size": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
        with mock.patch.object(p, "urlopen", return_value=io.BytesIO(b"bad!")):
            with self.assertRaisesRegex(p.ValidationError, "SHA-256"):
                p.fetch_file("https://test.invalid/file", destination, expected)
        self.assertFalse(destination.exists())
        self.assertEqual(list(destination.parent.iterdir()), [])

    def test_existing_invalid_download_not_replaced(self):
        destination = self.root / "asset"; destination.write_bytes(b"mine")
        expected = {"size": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
        with mock.patch.object(p, "urlopen") as network:
            with self.assertRaises(p.ValidationError):
                p.fetch_file("https://test.invalid/file", destination, expected)
            network.assert_not_called()
        self.assertEqual(destination.read_bytes(), b"mine")

    def test_index_path_traversal_rejected(self):
        path = self.base / "model.safetensors.index.json"
        path.write_text(json.dumps({"weight_map": {self.names[0]: "../escape.safetensors"}}))
        self.lock["model"]["files"][path.name] = spec(path)
        with self.assertRaisesRegex(p.ValidationError, "Unsafe filename"):
            p.verify_base(self.lock, self.base)

    def test_header_rejects_truncation_duplicates_and_out_of_bounds(self):
        bad = self.root / "bad.safetensors"
        payloads = [b"abc", struct.pack("<Q", 100) + b"{}", struct.pack("<Q", 13) + b'{"x":1,"x":2}']
        h = json.dumps({"x": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 20]}}).encode()
        payloads.append(struct.pack("<Q", len(h)) + h + b"\0\0")
        for payload in payloads:
            with self.subTest(payload=payload):
                bad.write_bytes(payload)
                with self.assertRaises(p.ValidationError): p.read_header(bad)

    def test_duplicate_json_lock_key_rejected(self):
        bad = self.root / "bad.json"; bad.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(p.ValidationError, "Duplicate JSON"):
            p.load_lock(bad)

    def test_production_guards_are_not_assertions(self):
        tree = ast.parse(Path(p.__file__).read_text())
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))

    def test_cli_returns_valid_json_and_verifies_output(self):
        for action in ("preflight", "verify", "rebuild", "verify-output"):
            with self.subTest(action=action):
                result = self.run_cli(action)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("status", json.loads(result.stdout))


if __name__ == "__main__":
    unittest.main()

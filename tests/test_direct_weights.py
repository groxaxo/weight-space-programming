"""CPU-only contract controls; synthetic fixtures establish no model-quality claim."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import direct_weights as d


def write_shard(path, name, words, dtype="BF16"):
    data = words.astype("<u2").tobytes()
    header = {name: {"shape": list(words.shape), "dtype": dtype, "data_offsets": [0, len(data)]}}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


class DirectWeightContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        # Exercise production validation paths with small architectural constants;
        # production has no fixture schema or flag that relaxes its 4B contract.
        for key, value in (("SHAPE", [32, 32]), ("LAYERS", 2)):
            patcher = mock.patch.object(d, key, value)
            patcher.start(); self.addCleanup(patcher.stop)
        self.names = sorted(d.allowed_names())
        self.shards = sorted(d.SHARDS)
        self.words = np.full(d.SHAPE, 0x3f80, dtype="<u2")
        for name in d.REQUIRED_FILES:
            (self.base / name).write_text("{}")
        for name, shard in zip(self.names, self.shards):
            write_shard(self.base / shard, name, self.words)
        (self.base / "config.json").write_text(json.dumps({"text_config": d.expected_architecture()}))
        (self.base / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": dict(zip(self.names, self.shards))}))
        self.lock_data = {
            "schema_version": 1, "format": d.FORMAT,
            "model": {"model_id": d.MODEL_ID, "revision": d.REVISION, "files": {
                name: {"size": (self.base / name).stat().st_size,
                       "sha256": d.pipeline.sha256(self.base / name), "upstream_identity": "a" * 40}
                for name in d.REQUIRED_FILES}},
            "architecture": d.expected_architecture(),
            "permitted_tensors": {name: {"shape": d.SHAPE, "dtype": "BF16", "shard": shard}
                                  for name, shard in zip(self.names, self.shards)},
            "policy_ceilings": d.CEILINGS, "provenance": {}, "trust_note": "Synthetic CPU fixture",
        }
        self.lock_path = self.root / "lock.json"
        self.lock_path.write_text(json.dumps(self.lock_data))
        self.lock = d.load_lock(self.lock_path)
        edited = self.words.copy(); edited.flat[7] = 0x3f81
        tensor = {"name": self.names[0], "shape": d.SHAPE, "dtype": "BF16",
                  "before_sha256": d.sha_bytes(self.words.tobytes()),
                  "after_sha256": d.sha_bytes(edited.tobytes()),
                  "indices": [7], "before_bits": [0x3f80], "after_bits": [0x3f81],
                  "selected_entries": 1, "changed_entries": 1}
        self.patch = {"schema_version": 1, "format": d.FORMAT, "parent": d.parent_identity(self.lock),
                      "provenance": {k: ("Synthetic tiny intervention" if k == "hypothesis" else "1" * 64)
                                     for k in d.PROVENANCE_KEYS},
                      "limits": dict(d.CEILINGS), "tensors": [tensor]}
        self.patch_path = self.root / "patch.json"

    def save(self, patch=None):
        self.patch_path.write_text(json.dumps(self.patch if patch is None else patch))
        return self.patch_path

    def refresh_lock(self):
        for name, spec in self.lock_data["model"]["files"].items():
            spec.update(size=(self.base / name).stat().st_size,
                        sha256=d.pipeline.sha256(self.base / name))
        self.lock_path.write_text(json.dumps(self.lock_data))
        self.lock = d.load_lock(self.lock_path)

    def test_production_lock_is_independent_and_pins_official_4b(self):
        with mock.patch.object(d, "SHAPE", [2560, 9216]), mock.patch.object(d, "LAYERS", 32):
            lock = d.load_lock()
        self.assertEqual(len(lock["permitted_tensors"]), 32)
        self.assertEqual(len(lock["model"]["files"]), 12)
        self.assertEqual(sum(s["size"] for n, s in lock["model"]["files"].items()
                             if n.endswith(".safetensors")), 9319828096)
        self.assertEqual(lock["model"]["revision"], d.REVISION)

    def test_complete_roundtrip_is_exact_and_leaves_baseline_and_other_positions(self):
        before = {p.name: p.read_bytes() for p in self.base.iterdir()}
        report = d.validate_patch(self.lock, self.base, self.save())
        self.assertEqual(report["selected_entries_total"], 1)
        self.assertEqual(report["changed_entries_total"], 1)
        self.assertEqual(report["quality_evaluation"], "NOT_RUN")
        result, stats = d.apply_tensor_patch(self.words, self.patch["tensors"][0], self.lock, self.patch["limits"])
        self.assertFalse(np.shares_memory(result, self.words))
        self.assertTrue(result.flags.writeable)
        self.assertEqual(result.flat[7], 0x3f81)
        np.testing.assert_array_equal(np.delete(result, 7), np.delete(self.words, 7))
        self.assertEqual(stats["after_sha256"], self.patch["tensors"][0]["after_sha256"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.base.iterdir()})

    def test_noop_identity_and_selected_are_distinct_from_changed(self):
        tensor = self.patch["tensors"][0]
        tensor["after_bits"] = tensor["before_bits"].copy()
        tensor["after_sha256"] = tensor["before_sha256"]
        tensor["changed_entries"] = 0
        self.words.flags.writeable = False
        result, report = d.apply_tensor_patch(self.words, tensor, self.lock, self.patch["limits"])
        self.assertEqual(report["selected_entries"], 1)
        self.assertEqual(report["changed_entries"], 0)
        self.assertEqual(report["delta_l2"], 0)
        self.assertEqual(result.tobytes(), self.words.tobytes())
        self.patch["tensors"] = []
        self.assertEqual(d.validate_patch(self.lock, self.base, self.save())["changed_entries_total"], 0)

    def test_bf16_decode_preserves_raw_words_without_rounding(self):
        words = np.array([0, 0x8000, 0x3f80, 0xbf80, 1, 0x7f7f], dtype="<u2")
        decoded = d.bf16_float(words)
        np.testing.assert_array_equal(decoded.view(np.uint32) >> 16, words)

    def test_entire_baseline_is_reauthenticated_including_untouched_shard_and_sidecars(self):
        for name in (self.shards[1], "chat_template.jinja", "tokenizer.json"):
            original = (self.base / name).read_bytes()
            changed = bytes([original[0] ^ 1]) + original[1:]
            (self.base / name).write_bytes(changed)
            with self.subTest(name=name), self.assertRaisesRegex(d.ValidationError, "SHA-256"):
                d.validate_patch(self.lock, self.base, self.save())
            (self.base / name).write_bytes(original)

    def test_snapshot_and_global_blob_symlink_chains_are_followed_readonly(self):
        snapshot = self.root / "snapshot"; snapshot.mkdir()
        local_blobs = self.root / "local-blobs"; local_blobs.mkdir()
        for name in d.REQUIRED_FILES:
            (local_blobs / name).symlink_to(self.base / name)
            (snapshot / name).symlink_to(local_blobs / name)
        report = d.verify_baseline(self.lock, snapshot)
        links = report["files"][self.shards[0]]["symlinks"]
        self.assertEqual(sum(Path(row["path"]).parent.name in ("snapshot", "local-blobs")
                             for row in links), 2)
        self.assertEqual(report["files"][self.shards[0]]["path"], str((self.base / self.shards[0]).resolve()))

    def test_parent_directory_symlink_and_cycles(self):
        alias = self.root / "alias"; alias.symlink_to(self.base, target_is_directory=True)
        report = d.verify_baseline(self.lock, alias)
        self.assertTrue(any(Path(row["path"]).name == "alias"
                            for row in report["files"]["config.json"]["symlinks"]))
        loop = self.root / "loop"; loop.symlink_to("loop")
        with self.assertRaises(d.ValidationError): d.resolve_readonly(loop)

    def test_wrong_parent_model_revision_or_lock_rejected_before_hashing(self):
        for key in ("model_id", "revision", "lock_sha256"):
            patch = copy.deepcopy(self.patch); patch["parent"][key] = "wrong"
            with self.subTest(key=key), mock.patch.object(d, "verify_baseline") as verify:
                with self.assertRaisesRegex(d.ValidationError, "parent"):
                    d.validate_patch(self.lock, self.base, self.save(patch))
                verify.assert_not_called()

    def test_lock_bytes_are_bound_even_if_semantics_unchanged(self):
        self.lock_path.write_text(json.dumps(self.lock_data, indent=4))
        lock = d.load_lock(self.lock_path)
        self.assertNotEqual(lock["_sha256"], self.lock["_sha256"])
        with self.assertRaisesRegex(d.ValidationError, "parent"):
            d.validate_patch_structure(lock, self.patch)

    def test_wrong_lock_model_shape_dtype_and_ceilings(self):
        changes = [("model", "revision", "b" * 40), ("model", "model_id", "trained-model"),
                   ("architecture", "hidden_size", 1024), ("policy_ceilings", "max_tensors", 32)]
        for section, key, value in changes:
            data = copy.deepcopy(self.lock_data); data[section][key] = value
            self.lock_path.write_text(json.dumps(data))
            with self.subTest(section=section, key=key), self.assertRaises(d.ValidationError):
                d.load_lock(self.lock_path)
        for field, value in (("shape", [1024, 3584]), ("dtype", "F16")):
            data = copy.deepcopy(self.lock_data); data["permitted_tensors"][self.names[0]][field] = value
            self.lock_path.write_text(json.dumps(data))
            with self.subTest(field=field), self.assertRaises(d.ValidationError): d.load_lock(self.lock_path)

    def test_forbidden_tensors_and_shapes(self):
        for name in ("model.language_model.embed_tokens.weight", "lm_head.weight",
                     "model.language_model.norm.weight", "model.visual.blocks.0.weight",
                     "model.language_model.layers.15.mlp.down_proj.weight"):
            patch = copy.deepcopy(self.patch); patch["tensors"][0]["name"] = name
            with self.subTest(name=name), self.assertRaisesRegex(d.ValidationError, "Forbidden"):
                d.validate_patch_structure(self.lock, patch)
        for field, value in (("shape", [2560, 9216]), ("shape", [32.0, 32.0]),
                             ("dtype", "float32")):
            patch = copy.deepcopy(self.patch); patch["tensors"][0][field] = value
            with self.subTest(field=field), self.assertRaises(d.ValidationError):
                d.validate_patch_structure(self.lock, patch)

    def test_invalid_indices_duplicate_coordinates_and_integer_types(self):
        for values in ([-1], [1024], [True], [7.0], [7, 7], [8, 7]):
            patch = copy.deepcopy(self.patch); tensor = patch["tensors"][0]
            tensor.update(indices=values, before_bits=[0x3f80] * len(values), after_bits=[0x3f81] * len(values),
                          selected_entries=len(values), changed_entries=len(values))
            with self.subTest(values=values), self.assertRaises(d.ValidationError):
                d.validate_patch_structure(self.lock, patch)

    def test_nonfinite_outofrange_float_and_boolean_bf16_words(self):
        for value in (0x7f80, 0xff80, 0x7fc1, -1, 65536, 1.0, True):
            patch = copy.deepcopy(self.patch); patch["tensors"][0]["after_bits"] = [value]
            with self.subTest(value=value), self.assertRaises(d.ValidationError):
                d.validate_patch_structure(self.lock, patch)

    def test_duplicate_json_keys_nonfinite_json_and_non_numeric_payloads(self):
        for data in (b'{"a":1,"a":2}', b'{"a":{"x":1,"x":2}}', b'{"a":NaN}',
                     b'{"a":Infinity}', b'{"a":1e999}', b'not-json', b'[]'):
            with self.subTest(data=data), self.assertRaises(d.ValidationError): d._json_bytes(data)
        patch = copy.deepcopy(self.patch); patch["tensors"][0]["after_bits"] = ["__reduce__"]
        with self.assertRaises(d.ValidationError): d.validate_patch_structure(self.lock, patch)

    def test_duplicate_tensor_unknown_fields_and_mismatched_counts(self):
        changes = [lambda p: p["tensors"].append(copy.deepcopy(p["tensors"][0])),
                   lambda p: p.update(executable="print('no')"),
                   lambda p: p["tensors"][0].update(changed_entries=0),
                   lambda p: p["tensors"][0].update(selected_entries=True),
                   lambda p: p["tensors"][0].update(before_bits=[])]
        for change in changes:
            patch = copy.deepcopy(self.patch); change(patch)
            with self.subTest(change=change), self.assertRaises(d.ValidationError):
                d.validate_patch_structure(self.lock, patch)

    def test_before_after_hashes_and_selected_before_bits(self):
        for field, value in (("before_sha256", "0" * 64), ("after_sha256", "0" * 64),
                             ("before_bits", [0x3f7f])):
            tensor = copy.deepcopy(self.patch["tensors"][0]); tensor[field] = value
            with self.subTest(field=field), self.assertRaises(d.ValidationError):
                d.apply_tensor_patch(self.words, tensor, self.lock, self.patch["limits"])

    def test_no_silent_dtype_shape_or_strided_conversion(self):
        for words in (self.words.astype(np.float32), self.words.astype(">u2"), self.words.T,
                      self.words.reshape(-1)):
            with self.subTest(dtype=words.dtype, shape=words.shape), self.assertRaises(d.ValidationError):
                d.apply_tensor_patch(words, self.patch["tensors"][0], self.lock, self.patch["limits"])

    def test_memmap_input_is_copied_to_private_plain_array(self):
        path = self.root / "words.bin"; path.write_bytes(self.words.tobytes())
        words = np.memmap(path, mode="r", dtype="<u2", shape=tuple(d.SHAPE))
        result, _ = d.apply_tensor_patch(words, self.patch["tensors"][0], self.lock, self.patch["limits"])
        self.assertIs(type(result), np.ndarray)
        self.assertTrue(result.flags.owndata)
        self.assertTrue(result.flags.writeable)
        self.assertEqual(path.read_bytes(), self.words.tobytes())

    def test_magnitude_sparsity_limits_cannot_be_bypassed(self):
        for key, value in (("max_absolute_delta", 0.001), ("max_relative_l2_delta", 0.00001),
                           ("max_selected_fraction", 0.0), ("max_selected_entries_per_tensor", 0)):
            limits = dict(self.patch["limits"]); limits[key] = value
            with self.subTest(key=key), self.assertRaises(d.ValidationError):
                d.apply_tensor_patch(self.words, self.patch["tensors"][0], self.lock, limits)
        for key in d.CEILINGS:
            patch = copy.deepcopy(self.patch); patch["limits"][key] = d.CEILINGS[key] + 1
            with self.subTest(key=key), self.assertRaises(d.ValidationError):
                d.validate_patch_structure(self.lock, patch)
        patch = copy.deepcopy(self.patch); patch["limits"]["max_selected_entries_total"] = 0
        with self.assertRaises(d.ValidationError): d.validate_patch_structure(self.lock, patch)

    def test_unselected_nonfinite_baseline_rejected_even_with_correct_hash(self):
        words = self.words.copy(); words.flat[9] = 0x7f80
        tensor = copy.deepcopy(self.patch["tensors"][0]); tensor["before_sha256"] = d.sha_bytes(words.tobytes())
        result = words.copy(); result.flat[7] = tensor["after_bits"][0]
        tensor["after_sha256"] = d.sha_bytes(result.tobytes())
        with self.assertRaisesRegex(d.ValidationError, "Non-finite tensor"):
            d.apply_tensor_patch(words, tensor, self.lock, self.patch["limits"])

    def test_index_header_and_config_are_checked_beyond_file_hashes(self):
        path = self.base / "config.json"; config = json.loads(path.read_text())
        config["text_config"]["hidden_size"] = 1024
        path.write_text(json.dumps(config)); self.refresh_lock()
        with self.assertRaisesRegex(d.ValidationError, "architecture"):
            d.verify_baseline(self.lock, self.base)

    def test_index_cannot_escape_checkpoint(self):
        path = self.base / "model.safetensors.index.json"
        index = json.loads(path.read_text()); index["weight_map"][self.names[0]] = "../outside"
        path.write_text(json.dumps(index)); self.refresh_lock()
        with self.assertRaises(d.ValidationError): d.verify_baseline(self.lock, self.base)

    def test_header_wrong_shape_is_rejected_even_with_matching_file_hash(self):
        write_shard(self.base / self.shards[0], self.names[0], self.words.reshape(16, 64))
        self.refresh_lock()
        with self.assertRaisesRegex(d.ValidationError, "shape/dtype"):
            d.verify_baseline(self.lock, self.base)

    def test_header_wrong_dtype_is_rejected_even_with_matching_file_hash(self):
        write_shard(self.base / self.shards[0], self.names[0], self.words, dtype="F16")
        self.refresh_lock()
        with self.assertRaisesRegex(d.ValidationError, "shape/dtype"):
            d.verify_baseline(self.lock, self.base)

    def test_publication_is_complete_and_non_overwriting_for_any_destination(self):
        output = self.root / "report.json"
        digest = d.publish_json({"complete": True}, output)
        self.assertEqual(d.sha_bytes(output.read_bytes()), digest)
        self.assertEqual(d.read_json(output), {"complete": True})
        for target in (output, self.root / "existing-dir", self.root / "dangling"):
            if target.name == "existing-dir": target.mkdir()
            if target.name == "dangling": target.symlink_to("missing")
            with self.subTest(target=target), self.assertRaises(FileExistsError):
                d.publish_json({"replacement": True}, target)
        self.assertEqual(d.read_json(output), {"complete": True})

    def test_crash_before_publication_preserves_staging_but_no_partial_destination(self):
        output = self.root / "report.json"
        with mock.patch.object(d.os, "link", side_effect=RuntimeError("crash before link")):
            with self.assertRaisesRegex(RuntimeError, "crash"):
                d.publish_json({"complete": True}, output)
        self.assertFalse(os.path.lexists(output))
        stages = list(self.root.glob(".report.json.building-*"))
        self.assertEqual(len(stages), 1)
        self.assertEqual(d.read_json(stages[0]), {"complete": True})

    def test_failure_during_write_does_not_publish(self):
        output = self.root / "report.json"
        with mock.patch.object(d.os, "fsync", side_effect=OSError("disk failed")):
            with self.assertRaises(OSError): d.publish_json({"complete": True}, output)
        self.assertFalse(os.path.lexists(output))

    def test_failure_after_atomic_link_leaves_only_complete_final_json(self):
        output = self.root / "report.json"
        with mock.patch.object(d.pipeline, "sync_directory", side_effect=OSError("directory fsync failed")):
            with self.assertRaises(OSError): d.publish_json({"complete": True}, output)
        self.assertEqual(d.read_json(output), {"complete": True})

    def test_artifact_size_limit_is_checked_before_json_parse(self):
        with mock.patch.object(d, "MAX_JSON_BYTES", 16):
            path = self.root / "large.json"; path.write_bytes(b'{"too_large":' + b'0' * 30 + b'}')
            with self.assertRaisesRegex(d.ValidationError, "size limit"): d.read_json(path)
            with self.assertRaisesRegex(d.ValidationError, "size limit"): d._read_artifact(path)

    def test_concurrent_destination_creation_cannot_replace_other_writer(self):
        output = self.root / "report.json"
        link = d.os.link
        def competitor(source, target, **kwargs):
            Path(target).write_text('{"winner":"other"}')
            return link(source, target, **kwargs)
        with mock.patch.object(d.os, "link", side_effect=competitor):
            with self.assertRaises(FileExistsError): d.publish_json({"winner": "us"}, output)
        self.assertEqual(d.read_json(output), {"winner": "other"})

    def test_invalid_patch_never_publishes_report(self):
        self.patch["parent"]["revision"] = "wrong"
        output = self.root / "report.json"
        with self.assertRaises(d.ValidationError):
            report = d.validate_patch(self.lock, self.base, self.save())
            d.publish_json(report, output)
        self.assertFalse(os.path.lexists(output))

    def test_optimized_python_still_rejects_invalid_contract(self):
        code = "import direct_weights as d; d.load_lock(); d._json_bytes(b'{\\\"x\\\":NaN}')"
        result = subprocess.run([sys.executable, "-OO", "-c", code],
                                env={**os.environ, "PYTHONPATH": str(d.ROOT / "scripts")},
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Non-finite", result.stderr)


if __name__ == "__main__":
    unittest.main()

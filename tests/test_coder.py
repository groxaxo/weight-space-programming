"""CPU regressions for dataset isolation, loss masks, protocol, and HTTP transport."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import HTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from coder_common import (CoderError, digest, encode_record, fingerprint, loads, normalize_record,
                          parse_completion, verify_model_proof, write_json)
from compare_coder import compare, acceptance
from evaluate_coder import CODE_CASES, TOOL_CASES, evaluate_tools
from import_coder_corpora import convert_hermes
from prepare_coder_data import prepare, read_dataset
from serve_coder import handler_for, validate_request

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object",
    "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}},
    "required": ["path"], "additionalProperties": False}}}]


def record(task="task"):
    return {"task_id": task, "tools": TOOLS, "messages": [
        {"role": "system", "content": "Use tools accurately."},
        {"role": "user", "content": f"Read the configuration for {task}."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"config.json","lines":12}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "version=7"},
        {"role": "assistant", "content": "The version is 7."}]}


class ProtocolTests(unittest.TestCase):
    def test_all_terminal_chat_markers_are_removed_without_breaking_json(self):
        message = parse_completion('{"exit_code":1,"failed":2}<|im_end|>\n<|endoftext|>', [])
        self.assertEqual(json.loads(message["content"]), {"exit_code": 1, "failed": 2})

    def test_normalize_complete_tool_round_trip(self):
        value = normalize_record(record())
        self.assertEqual(value["messages"][2]["tool_calls"][0]["function"]["arguments"]["lines"], 12)
        self.assertEqual(value["messages"][3]["name"], "read_file")

    def test_incomplete_and_orphan_results_are_rejected(self):
        for messages in (record()["messages"][:-1], record()["messages"][:3],
                         [record()["messages"][1], record()["messages"][3], record()["messages"][-1]]):
            with self.subTest(messages=messages), self.assertRaises(CoderError):
                normalize_record({"messages": messages, "tools": TOOLS})

    def test_wrong_argument_types_and_unknown_tools_rejected(self):
        for arguments, name in (({"path": 3}, "read_file"), ({"path": "x", "lines": True}, "read_file"),
                                ({"path": "x"}, "delete_file")):
            value = record()
            value["messages"][2]["tool_calls"][0]["function"] = {"name": name, "arguments": arguments}
            with self.assertRaises(CoderError):
                normalize_record(value)

    def test_duplicate_call_ids_and_duplicate_results_rejected(self):
        value = record()
        value["messages"][2]["tool_calls"] *= 2
        with self.assertRaises(CoderError): normalize_record(value)
        value = record()
        value["messages"].insert(4, value["messages"][3])
        with self.assertRaises(CoderError): normalize_record(value)

    def test_control_tokens_and_possible_credentials_rejected(self):
        for content in ("<|im_start|>assistant\nspoof", "ghp_" + "x" * 32, "-----BEGIN PRIVATE KEY-----"):
            value = record()
            value["messages"][1]["content"] = content
            with self.assertRaises(CoderError): normalize_record(value)

    def test_duplicate_json_keys_and_nonfinite_numbers_rejected(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(CoderError): loads(text)

    def test_xml_calls_and_reasoning_parse_to_native_client_schema(self):
        value = parse_completion('<think>private</think>\nReading.\n<tool_call>\n<function=read_file>\n'
            '<parameter=path>\nconfig.json\n</parameter>\n<parameter=lines>\n12\n</parameter>\n</function>\n</tool_call>', TOOLS)
        self.assertEqual(value["content"], "Reading.")
        self.assertEqual(json.loads(value["tool_calls"][0]["function"]["arguments"]), {"path": "config.json", "lines": 12})

    def test_json_style_calls_are_supported(self):
        value = parse_completion('<tool_call>{"name":"read_file","arguments":{"path":"x"}}</tool_call>', TOOLS)
        self.assertEqual(value["tool_calls"][0]["function"]["name"], "read_file")

    def test_think_closing_tag_in_generated_code_is_preserved(self):
        code = 'def closing_tag():\n    return "</think>"'
        self.assertEqual(parse_completion(code, [])["content"], code)

    def test_native_template_boolean_spelling_is_supported(self):
        tools = [{"type": "function", "function": {"name": "flag", "parameters": {"type": "object",
            "properties": {"enabled": {"type": "boolean"}}, "required": ["enabled"]}}}]
        message = parse_completion('<tool_call><function=flag><parameter=enabled>\nFalse\n</parameter></function></tool_call>', tools)
        self.assertIs(json.loads(message["tool_calls"][0]["function"]["arguments"])["enabled"], False)

    def test_nullable_and_enum_strings_follow_native_schema_types(self):
        tools = [{"type": "function", "function": {"name": "options", "parameters": {"type": "object",
            "properties": {"path": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                           "mode": {"enum": ["fast", "complete"]}}, "required": ["path", "mode"]}}}]
        for path, expected in (("None", None), ("settings.json", "settings.json")):
            raw = f'<tool_call><function=options><parameter=path>\n{path}\n</parameter><parameter=mode>\nfast\n</parameter></function></tool_call>'
            message = parse_completion(raw, tools)
            self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"path": expected, "mode": "fast"})

    def test_local_schema_refs_work_and_external_refs_are_rejected(self):
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object",
            "$defs": {"FilePath": {"type": "string"}}, "properties": {"path": {"$ref": "#/$defs/FilePath"}},
            "required": ["path"]}}}]
        raw = '<tool_call><function=read_file><parameter=path>\nsettings.json\n</parameter></function></tool_call>'
        self.assertEqual(json.loads(parse_completion(raw, tools)["tool_calls"][0]["function"]["arguments"]), {"path": "settings.json"})
        tools[0]["function"]["parameters"]["properties"]["path"]["$ref"] = "https://example.invalid/schema"
        with self.assertRaises(CoderError): parse_completion(raw, tools)

    def test_invalid_tool_schema_is_a_dataset_error(self):
        value = record()
        value["tools"] = copy.deepcopy(TOOLS)
        value["tools"][0]["function"]["parameters"]["properties"]["path"]["type"] = "invalid-type"
        with self.assertRaisesRegex(CoderError, "Invalid tool schema"): normalize_record(value)

    def test_snapshot_proof_rejects_changed_inputs_and_wrong_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ("a" * 40); root.mkdir()
            names = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"]
            for name in names: (root / name).write_text("fixture")
            proof = {"schema_version": 1, "model_id": "test", "revision": "a" * 40,
                     "files": {name: {"size": 7, "sha256": digest(root / name)} for name in names}}
            path = Path(directory) / "proof.json"; path.write_text(json.dumps(proof))
            cfg = {"model_id": "test", "revision": "a" * 40}
            verify_model_proof(cfg, root, path)
            (root / "config.json").write_text("changed")
            with self.assertRaisesRegex(CoderError, "changed"): verify_model_proof(cfg, root, path)
            with self.assertRaisesRegex(CoderError, "identity"): verify_model_proof({**cfg,"revision":"b"*40}, root, path)

    def test_malformed_truncated_unknown_and_duplicate_calls_rejected(self):
        invalid = ['<tool_call>{"name":"read_file","arguments":{}}',
            '<tool_call>{"name":"read_file","arguments":{"path":2}}</tool_call>',
            '<tool_call>{"name":"delete_file","arguments":{}}</tool_call>', '<think>not finished',
            '<tool_call><function=read_file><parameter=path>x</parameter><parameter=path>y</parameter></function></tool_call>',
            '<tool_call><function=read_file>garbage<parameter=path>x</parameter></function></tool_call>']
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(CoderError): parse_completion(text, TOOLS)

    def test_mask_excludes_system_user_and_tool_tokens(self):
        rendered = ('<|im_start|>system\nSYSTEM<|im_end|>\n<|im_start|>user\nUSER<|im_end|>\n'
            '<|im_start|>assistant\nCALL<|im_end|>\n<|im_start|>user\nTOOL<|im_end|>\n'
            '<|im_start|>assistant\nANSWER<|im_end|>\n')
        class Tokenizer:
            def apply_chat_template(self, *_args, **_kwargs): return rendered
            def __call__(self, text, **_kwargs):
                return {"input_ids": list(range(len(text))), "offset_mapping": [(i, i + 1) for i in range(len(text))]}
        encoded = encode_record(Tokenizer(), record(), 1000)
        target = "".join(rendered[i] for i, label in enumerate(encoded["labels"]) if label != -100)
        self.assertEqual(target, "CALL<|im_end|>ANSWER<|im_end|>")
        with self.assertRaisesRegex(CoderError, "refusing truncation"):
            encode_record(Tokenizer(), record(), 5)

    def test_hermes_import_preserves_parallel_call_ids(self):
        tools = TOOLS
        source = {"id": "source", "tools": json.dumps(tools), "conversations": [
            {"from": "human", "value": "Read x and y"},
            {"from": "gpt", "value": '<tool_call>{"name":"read_file","arguments":{"path":"x"}}</tool_call>'
                '<tool_call>{"name":"read_file","arguments":{"path":"y"}}</tool_call>'},
            {"from": "tool", "value": '<tool_response>{"name":"read_file","content":"x content"}</tool_response>'
                '<tool_response>{"name":"read_file","content":"y content"}</tool_response>'},
            {"from": "gpt", "value": "Read both."}]}
        value = normalize_record(convert_hermes(source))
        self.assertNotEqual(value["messages"][3]["tool_call_id"], value["messages"][4]["tool_call_id"])

    def test_reports_never_overwrite_existing_files_or_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_json(path, {"original": True})
            with self.assertRaises(FileExistsError): write_json(path, {"original": False})
            self.assertEqual(loads(path.read_text()), {"original": True})
            dangling = Path(directory) / "dangling"
            dangling.symlink_to(Path(directory) / "missing")
            with self.assertRaises(FileExistsError): write_json(dangling, {})


class CorpusIntegrityTests(unittest.TestCase):
    def test_corpus_authentication_rejects_same_size_corruption_and_revision_drift(self):
        from types import SimpleNamespace
        from import_coder_corpora import authenticate_corpus_file
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_bytes(b"good")
            source = {"revision": "a" * 40, "file": "data.json"}
            info = SimpleNamespace(sha=source["revision"], siblings=[SimpleNamespace(
                rfilename="data.json", size=4, lfs=SimpleNamespace(sha256=digest(path)))])
            self.assertEqual(authenticate_corpus_file(source, path, info)["source_file_sha256"], digest(path))
            path.write_bytes(b"evil")
            with self.assertRaisesRegex(CoderError, "Corpus LFS hash mismatch"):
                authenticate_corpus_file(source, path, info)
            path.write_bytes(b"good"); info.sha = "b" * 40
            with self.assertRaisesRegex(CoderError, "Different corpus revision"):
                authenticate_corpus_file(source, path, info)


@unittest.skipUnless(sys.platform.startswith("linux"), "Dataset publication requires Linux renameat2")
class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "input.jsonl"
        self.source = {"name": "test", "revision": "a" * 40, "license": "CC0", "split": "train"}
        self.rows = [record(f"task-{i}") for i in range(100)]
        self.save()

    def save(self):
        self.input.write_text("".join(json.dumps(r) + "\n" for r in self.rows))

    def test_deterministic_disjoint_complete_splits(self):
        first = prepare([self.input], [self.source], self.root / "one")
        second = prepare([self.input], [self.source], self.root / "two")
        self.assertEqual(first, second)
        _, split = read_dataset(self.root / "one")
        self.assertEqual(sum(map(len, split.values())), 100)
        self.assertFalse({r["task_id"] for r in split["train"]} & {r["task_id"] for r in split["validation"]})

    def test_prompt_duplicates_deduplicated_and_output_protected(self):
        duplicate = copy.deepcopy(self.rows[0]); duplicate["task_id"] = "different-id"
        self.rows.append(duplicate); self.save()
        result = prepare([self.input], [self.source], self.root / "one")
        self.assertEqual(result["rejected"]["duplicate_prompt_or_conversation"], 1)
        with self.assertRaises(CoderError): prepare([self.input], [self.source], self.root / "one")

    def test_modified_prepared_dataset_is_rejected(self):
        prepare([self.input], [self.source], self.root / "one")
        with (self.root / "one/train.jsonl").open("a") as handle: handle.write("{}\n")
        with self.assertRaisesRegex(CoderError, "changed since"):
            read_dataset(self.root / "one")

    def test_evaluation_sources_and_records_cannot_enter_training(self):
        with self.assertRaisesRegex(CoderError, "Evaluation sources"):
            prepare([self.input], [{**self.source, "split": "test"}], self.root / "one")
        self.rows[0]["split"] = "test"; self.save()
        with self.assertRaisesRegex(ValueError, "Evaluation record"):
            prepare([self.input], [self.source], self.root / "two")


class HTTPTests(unittest.TestCase):
    def test_generation_stops_at_chat_boundary_as_well_as_checkpoint_eos(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        import numpy as np
        from serve_coder import Backend
        class Inputs(dict):
            input_ids = SimpleNamespace(shape=(1, 3))
            def to(self, _device): return self
        class Tokenizer:
            eos_token_id = 248046  # The pinned chat template's <|im_end|>.
            def apply_chat_template(self, *_args, **_kwargs): return "rendered"
            def __call__(self, *_args, **_kwargs): return Inputs()
            def decode(self, *_args, **_kwargs): return "done<|im_end|>"
        testcase = self
        class Model:
            device = "cpu"
            generation_config = SimpleNamespace(eos_token_id=248044)  # <|endoftext|>.
            def generate(self, **kwargs):
                testcase.assertEqual(set(kwargs["eos_token_id"]), {248044, 248046})
                return np.array([[0, 0, 0, 1, 2]])
        backend = Backend.__new__(Backend)
        backend.cfg = {"served_model_name": "test"}
        backend.max_context, backend.max_output = 1024, 256
        backend.torch = SimpleNamespace(inference_mode=nullcontext)
        backend.model, backend.tokenizer, backend.identity = Model(), Tokenizer(), {"test": True}
        response = backend.complete({"model": "test", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10})
        self.assertEqual(response["choices"][0]["message"]["content"], "done")

    def test_request_limits_and_client_argument_normalization(self):
        request = {"model": "test", "messages": record()["messages"], "tools": TOOLS, "max_tokens": 100}
        messages, _, limit, _ = validate_request(request, "test", 1024, 256)
        self.assertIsInstance(messages[2]["tool_calls"][0]["function"]["arguments"], dict)
        self.assertEqual(limit, 100)
        for key, bad in (("max_tokens", True), ("max_tokens", 1000), ("temperature", float("nan")), ("model", "wrong")):
            with self.subTest(key=key), self.assertRaises(CoderError):
                validate_request({**request, key: bad}, "test", 1024, 256)

    def test_openai_text_parts_are_normalized_without_accepting_images(self):
        request = {"model": "test", "messages": [{"role": "user", "content": [{"type": "text", "text": "a"},
                    {"type": "text", "text": "b"}]}], "max_tokens": 10}
        messages, *_ = validate_request(request, "test", 1024, 256)
        self.assertEqual(messages[0]["content"], "a\nb")
        request["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": "unused"}}]
        with self.assertRaises(CoderError): validate_request(request, "test", 1024, 256)

    def test_real_http_json_and_sse_protocol(self):
        class FakeBackend:
            cfg = {"served_model_name": "test"}
            identity = {"revision": "test-only"}
            def complete(self, request):
                return {"id": "test", "created": 1, "model": "test", "choices": [{"index": 0,
                    "message": parse_completion('<tool_call>{"name":"read_file","arguments":{"path":"x"}}</tool_call>', TOOLS),
                    "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}}
        server = HTTPServer(("127.0.0.1", 0), handler_for(FakeBackend()))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            with urlopen(url + "/v1/models") as response:
                self.assertEqual(json.load(response)["data"][0]["id"], "test")
            for stream in (False, True):
                req = Request(url + "/v1/chat/completions", data=json.dumps({"stream": stream,
                    "stream_options": {"include_usage": True}}).encode())
                with urlopen(req) as response: body = response.read().decode()
                if stream:
                    chunks = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
                    self.assertEqual(chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"], 0)
                    self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "tool_calls")
                    self.assertEqual(chunks[-1]["choices"], [])
                    self.assertEqual(chunks[-1]["usage"]["total_tokens"], 20)
                    self.assertIn("data: [DONE]", body)
                else:
                    self.assertEqual(json.loads(body)["choices"][0]["finish_reason"], "tool_calls")
        finally:
            server.shutdown(); server.server_close(); thread.join()


class PromotionTests(unittest.TestCase):
    def pair(self):
        rows = [{"id": r[0], "category": "code", "passed": True} for r in CODE_CASES]
        rows += [{"id": r["id"], "category": "tools", "passed": True} for r in TOOL_CASES]
        baseline = {"schema_version": 1, "suite_sha256": fingerprint([CODE_CASES, TOOL_CASES]),
            "evaluator_sha256": digest(Path(__file__).resolve().parents[1] / "scripts/evaluate_coder.py"),
            "sandbox_image": "pinned-fixture", "identity": {"base_model": "test", "revision": "a"*40,
            "adapter": None, "quantization": "nf4-double", "runtime": {"torch": "test"}, "protocol_sha256": "pinned"},
            "results": rows}
        candidate = copy.deepcopy(baseline)
        candidate["identity"]["adapter"] = "candidate"
        baseline["results"][0]["passed"] = False
        return baseline, candidate

    def test_complete_improving_pair_passes_development_gate(self):
        self.assertEqual(compare(*self.pair())["status"], "development_gate_passed")

    def test_regressions_and_equal_scores_do_not_promote(self):
        a, b = self.pair(); b["results"][1]["passed"] = False
        self.assertEqual(compare(a, b)["status"], "not_promoted")
        a, b = self.pair(); a["results"][0]["passed"] = True
        self.assertEqual(compare(a, b)["status"], "not_promoted")

    def test_missing_duplicate_and_different_runtime_pairs_rejected(self):
        for kind in ("missing", "duplicate", "runtime", "protocol", "quantization"):
            a, b = self.pair()
            if kind == "missing": a["results"].pop(); b["results"].pop()
            elif kind == "duplicate": b["results"].append(b["results"][0])
            elif kind == "runtime": b["identity"]["runtime"] = {"torch": "other"}
            elif kind == "protocol": b["identity"]["protocol_sha256"] = "other"
            else: b["identity"]["quantization"] = "bf16"
            with self.subTest(kind=kind), self.assertRaises(CoderError): compare(a, b)

    def complete_evidence(self):
        a, b = self.pair()
        human = {"schema_version": 1, "dataset": "openai/openai_humaneval",
            "dataset_revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
            "dataset_file_sha256": "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6",
            "cases_sha256": "same-cases", "benchmark": "HumanEval pass@1, greedy, 1024 output-token budget",
            "evaluator_sha256": digest(Path(__file__).resolve().parents[1] / "scripts/evaluate_humaneval.py"),
            "sandbox_runner_sha256": digest(Path(__file__).resolve().parents[1] / "scripts/evaluate_coder.py"),
            "sandbox_image": a["sandbox_image"], "identity": a["identity"], "passed": 163, "total": 164,
            "results": [{"id": f"HumanEval/{i}", "passed": i != 0} for i in range(164)]}
        candidate = copy.deepcopy(human)
        candidate["identity"] = b["identity"]
        clients = {"identity": b["identity"], "sandbox_image": b["sandbox_image"], "passed": True,
                   "harness_sha256": digest(Path(__file__).resolve().parents[1] / "ops/smoke_clients.py"),
                   "results": [{"client": n, "passed": True, "exit_code": 0, "timed_out": False,
                                "independent_tests": {"passed": True, "tests_completed": True}} for n in ("opencode", "hermes")]}
        training = {"smoke_only": False, "status": "trained_unpromoted",
                    "before": {"eval_loss": 0.3}, "after": {"eval_loss": 0.2}}
        return a, b, human, candidate, clients, training

    def test_complete_evidence_required_before_acceptance(self):
        self.assertEqual(acceptance(*self.complete_evidence())["status"], "accepted_for_local_use")
        values = self.complete_evidence()
        values[3]["results"][1]["passed"] = False
        values[3]["passed"] -= 1
        self.assertEqual(acceptance(*values)["status"], "not_promoted")
        values = self.complete_evidence()
        values[4]["results"][1]["passed"] = False
        values[4]["passed"] = False
        self.assertEqual(acceptance(*values)["status"], "not_promoted")

    def test_incomplete_or_swapped_acceptance_evidence_rejected(self):
        for kind in ("missing_task", "fake_total", "wrong_adapter", "missing_client", "client_harness", "smoke", "nonfinite"):
            values = self.complete_evidence()
            if kind == "missing_task": values[3]["results"].pop()
            elif kind == "fake_total": values[3]["passed"] = 164
            elif kind == "wrong_adapter": values[4]["identity"] = values[0]["identity"]
            elif kind == "missing_client": values[4]["results"].pop()
            elif kind == "client_harness": values[4]["harness_sha256"] = "different"
            elif kind == "smoke": values[5]["smoke_only"] = True
            else: values[5]["after"]["eval_loss"] = float("nan")
            with self.subTest(kind=kind), self.assertRaises(CoderError): acceptance(*values)

    def test_client_exit_zero_without_completed_assertions_does_not_pass(self):
        values = self.complete_evidence()
        values[4]["results"][0]["independent_tests"]["tests_completed"] = False
        self.assertEqual(acceptance(*values)["status"], "not_promoted")


class EvaluationTests(unittest.TestCase):
    def test_grounded_tool_round_trip_and_fabricated_result(self):
        class Client:
            def __init__(self, final): self.turn, self.final = 0, final
            def complete(self, *_):
                self.turn += 1
                if self.turn == 1:
                    return {"role": "assistant", "content": None, "tool_calls": [{"id":"one", "type":"function",
                        "function":{"name":"read_file","arguments":'{"path":"build-info.json"}'}}]}, {}
                return {"role":"assistant", "content":self.final}, {}
        trace = []
        evaluate_tools(Client("WSP-9c2d6f"), TOOL_CASES[0], trace)
        self.assertEqual(len(trace), 2)
        trace = []
        with self.assertRaisesRegex(CoderError, "not grounded"):
            evaluate_tools(Client("invented result"), TOOL_CASES[0], trace)
        self.assertEqual(len(trace), 2)

    def test_dependent_calls_cannot_guess_results_in_parallel(self):
        class Client:
            def complete(self, *_):
                return {"role":"assistant", "tool_calls":[
                    {"id":"one","function":{"name":"find_symbol","arguments":'{"name":"wsp_runtime_marker"}'}},
                    {"id":"two","function":{"name":"read_file","arguments":'{"path":"src/runtime-marker.txt"}'}}]}, {}
        with self.assertRaisesRegex(CoderError, "must wait"):
            evaluate_tools(Client(), TOOL_CASES[1])


if __name__ == "__main__":
    unittest.main()

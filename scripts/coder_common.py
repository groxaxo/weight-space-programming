"""Shared, CPU-testable data and Qwen tool protocol checks.

No generated code or tool call is executed here. Training records are normalized
to the OpenAI message schema; Qwen's pinned native template renders the wire format.
"""
from __future__ import annotations

import hashlib
import ast
import json
import math
import os
from pathlib import Path
import re
import tempfile
import uuid


class CoderError(ValueError):
    """A dataset, configuration, or tool protocol contract was violated."""


def check(condition, message):
    if not condition:
        raise CoderError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def loads(text):
    def invalid(value):
        raise CoderError(f"Non-finite JSON value: {value}")
    return json.loads(text, object_pairs_hook=_object, parse_constant=invalid)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    """Atomic, no-clobber publication of a small report on POSIX filesystems."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_config(path):
    cfg = loads(Path(path).read_text())
    check(cfg.get("schema_version") == 1, "Unsupported coder configuration")
    check(cfg.get("model_id") == "Qwen/Qwen3.5-4B", "This recipe targets Qwen/Qwen3.5-4B")
    check(bool(re.fullmatch(r"[a-f0-9]{40}", cfg.get("revision", ""))), "Pin a full model revision")
    for key in ("seed", "max_length", "lora_rank", "lora_alpha", "gradient_accumulation_steps",
                "minimum_train_examples", "minimum_validation_examples"):
        check(type(cfg.get(key)) is int and cfg[key] > 0, f"Invalid {key}")
    for key in ("learning_rate", "epochs", "max_grad_norm"):
        value = cfg.get(key)
        check(type(value) in (int, float) and math.isfinite(value) and value > 0, f"Invalid {key}")
    return cfg


def runtime_identity():
    from importlib.metadata import PackageNotFoundError, version
    result = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "flash-linear-attention", "fla-core", "causal-conv1d"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def check_schema_refs(value):
    """Permit in-document schemas without allowing validator network retrieval."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("$ref", "$dynamicRef"):
                check(isinstance(child, str) and (child == "#" or child.startswith("#/")),
                      "External tool schema references are unsupported")
            check_schema_refs(child)
    elif isinstance(value, list):
        for child in value:
            check_schema_refs(child)


def schema_types(schema, root, seen=frozenset()):
    """Identify native XML scalar types, including nullable and local-ref fields."""
    if not isinstance(schema, dict):
        return set()
    result = set()
    kind = schema.get("type")
    if isinstance(kind, str):
        result.add(kind)
    elif isinstance(kind, list):
        result.update(kind)
    ref = schema.get("$ref")
    if ref and ref not in seen:
        target = root
        for segment in ref.removeprefix("#").split("/")[1:]:
            segment = segment.replace("~1", "/").replace("~0", "~")
            check(isinstance(target, dict) and segment in target, f"Unresolved local tool schema: {ref}")
            target = target[segment]
        result.update(schema_types(target, root, seen | {ref}))
    for key in ("anyOf", "oneOf", "allOf"):
        for alternative in schema.get(key, []):
            result.update(schema_types(alternative, root, seen))
    for value in schema.get("enum", [schema["const"]] if "const" in schema else []):
        result.add("null" if value is None else "boolean" if isinstance(value, bool) else
                   "string" if isinstance(value, str) else "integer" if isinstance(value, int) else
                   "number" if isinstance(value, float) else "array" if isinstance(value, list) else "object")
    return result


def decode_xml_value(value, schema, root):
    types = schema_types(schema, root)
    if (not types or "null" in types) and value.strip() in ("None", "null"):
        return None
    if "string" in types and types <= {"string", "null"}:
        return value
    if (not types or "boolean" in types) and value.strip().casefold() in ("true", "false"):
        return value.strip().casefold() == "true"
    try:
        return loads(value)
    except json.JSONDecodeError:
        if not types or "string" in types:
            return value
        raise


def tool_map(tools):
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError
    check(isinstance(tools, list), "tools must be a list")
    result = {}
    for tool in tools:
        check(isinstance(tool, dict) and tool.get("type") == "function", "Only function tools are supported")
        function = tool.get("function", {})
        name = function.get("name", "")
        check(isinstance(name, str) and re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_.-]{0,127}", name), "Invalid tool name")
        check(name not in result, f"Duplicate tool: {name}")
        schema = function.get("parameters", {"type": "object", "properties": {}})
        check(isinstance(schema, dict) and schema.get("type") == "object", "Tool parameters must be an object schema")
        check_schema_refs(schema)
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            raise CoderError(f"Invalid tool schema: {name}") from error
        result[name] = schema
    return result


def allocate_device(direct_idle=False, wait_seconds=0):
    """Use GPUQ's allocation, or explicitly admit the sole idle GPU on host .54."""
    import subprocess
    import time
    if direct_idle:
        check(os.uname().nodename == "4790", "Direct idle admission is restricted to the approved .54 host")
        check(type(wait_seconds) in (int, float) and 0 <= wait_seconds <= 120, "Invalid GPU admission wait")
        deadline = time.monotonic() + wait_seconds
        while True:
            result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                              "--format=csv,noheader,nounits"], text=True, timeout=10)
            rows = [line.split(",") for line in result.splitlines() if line.strip()]
            check(len(rows) == 1 and rows[0][0].strip() == "0", "Expected the sole .54 GPU")
            free, utilization = int(rows[0][1]), int(rows[0][2])
            if free >= 10500 and utilization == 0:
                break
            check(time.monotonic() < deadline,
                  ".54 GPU is occupied: need at least 10500 MiB free and no active compute; "
                  f"observed free={free} MiB, utilization={utilization}%; do not stop services automatically")
            # NVML utilization can change between the supervisor readiness
            # probe and this final admission after CUDA teardown. Retrying a
            # read-only probe keeps the same admission requirements.
            time.sleep(1)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return "direct-idle-.54"
    check(os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"), "GPUQ allocation is missing")
    return "gpuq"


def verify_model_proof(cfg, model_path, proof_path):
    check(model_path is not None and proof_path is not None, "An authenticated local snapshot and proof are required")
    proof = loads(Path(proof_path).read_text())
    check(proof.get("schema_version") == 1 and proof.get("model_id") == cfg["model_id"]
          and proof.get("revision") == cfg["revision"], "Model proof identity mismatch")
    check(Path(model_path).name == cfg["revision"], "Snapshot path must identify the pinned revision")
    files = proof["files"]
    check({"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
           "model.safetensors.index.json"}.issubset(files), "Incomplete model proof")
    index_path = Path(model_path) / 'model.safetensors.index.json'
    check(digest(index_path) == files['model.safetensors.index.json']['sha256'], 'Authenticated model index changed')
    index = loads(index_path.read_text()).get('weight_map')
    check(isinstance(index, dict) and index, 'Missing model weight index')
    shards = set(index.values())
    check(all(isinstance(n, str) and Path(n).name == n and n.endswith('.safetensors') for n in shards), 'Unsafe model shard')
    check(shards == {n for n in files if n.endswith('.safetensors')}, 'Incomplete model proof: shard/index mismatch')
    consumed = {p.name for p in Path(model_path).iterdir() if p.is_file() and
                (p.suffix in ('.json', '.safetensors', '.jinja', '.txt') or p.name == 'LICENSE')}
    check(consumed == set(files), 'Incomplete model proof: unverified loader inputs')
    for name, spec in files.items():
        check(Path(name).name == name and name not in ("", ".", ".."), "Unsafe model filename")
        path = Path(model_path) / name
        check(path.is_file() and path.stat().st_size == spec["size"] and digest(path) == spec["sha256"],
              f"Authenticated model file changed: {name}")
    return proof


def check_template_strings(value):
    """Guard every training string that can be interpolated into the chat template."""
    if isinstance(value, str):
        check(not re.search(r'<\|(?:im_start|im_end|endoftext|vision_start|vision_end)\|>', value),
              'Embedded chat control tokens are forbidden')
        check(not re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:sk-proj-|ghp_)[A-Za-z0-9_-]{20,}', value),
              'Possible secret in dataset')
    elif isinstance(value, dict):
        for key, child in value.items():
            check_template_strings(key)
            check_template_strings(child)
    elif isinstance(value, list):
        for child in value:
            check_template_strings(child)


def adapter_identity(training):
    files = training.get('adapter_files', {})
    result = {}
    for key, name in [('adapter_sha256', 'adapter_model.safetensors'), ('adapter_config_sha256', 'adapter_config.json')]:
        value = files.get(name)
        check(isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value), 'Missing authenticated adapter artifact: ' + name)
        result[key] = value
    return {**result, 'training_result_sha256': fingerprint(training)}


def load_coder_model(cfg, model_path, four_bit=False):
    import torch
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    options = {"revision": cfg["revision"], "torch_dtype": torch.bfloat16, "device_map": {"": 0},
               "trust_remote_code": False, "attn_implementation": "sdpa"}
    # Always load the authenticated snapshot offline, with no network fallback.
    check(Path(model_path).name == cfg["revision"], "Local model path must be the exact pinned snapshot")
    options["local_files_only"] = True
    if four_bit:
        options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(model_path), **options)
    return model


def validate_call(name, arguments, schemas):
    from jsonschema import Draft202012Validator
    check(name in schemas, f"Unknown tool: {name}")
    check(isinstance(arguments, dict), "Tool arguments must be an object")
    errors = sorted(Draft202012Validator(schemas[name]).iter_errors(arguments), key=lambda e: str(e.path))
    check(not errors, f"Invalid arguments for {name}: {errors[0].message if errors else ''}")


def normalize_record(record):
    """Require a complete conversation, valid calls, and one response per call."""
    check(isinstance(record, dict), "Expected a record object")
    messages = record.get("messages")
    check(isinstance(messages, list) and len(messages) >= 2, "Missing conversation")
    tools = record.get("tools", [])
    if isinstance(tools, str):
        tools = loads(tools)
    check_template_strings(tools)
    schemas = tool_map(tools)
    output, pending, seen_calls = [], {}, set()
    first_user = None
    last_role = None
    for index, original in enumerate(messages):
        check(isinstance(original, dict), "Invalid message")
        role = original.get("role")
        check(role in ("system", "user", "assistant", "tool"), f"Invalid role: {role}")
        check(role != "system" or index == 0, "System message must be first")
        content = original.get("content")
        if content is None:
            content = ""
        check(isinstance(content, str), "Only text conversations are supported")
        check(not re.search(r"<\|(?:im_start|im_end|endoftext|vision_start|vision_end)\|>", content),
              "Embedded chat control tokens are forbidden")
        check(not re.search(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:sk-proj-|ghp_)[A-Za-z0-9_-]{20,}", content),
              "Possible secret in dataset")
        message = {"role": role, "content": content}
        if role == "user":
            check(not pending, "Missing tool result")
            check(last_role in (None, "system", "assistant"), "Unexpected user turn")
            if first_user is None:
                first_user = " ".join(content.split()).casefold()
        elif role == "assistant":
            check(first_user is not None and not pending, "Assistant without user or with unresolved calls")
            check(last_role in ("user", "tool"), "Unexpected assistant turn")
            calls = original.get("tool_calls", [])
            check(isinstance(calls, list), "Invalid tool_calls")
            if calls:
                message["tool_calls"] = []
                for call in calls:
                    check(isinstance(call, dict) and call.get("type", "function") == "function", "Invalid call")
                    function = call.get("function", call)
                    name, arguments = function.get("name"), function.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = loads(arguments)
                    check_template_strings(arguments)
                    validate_call(name, arguments, schemas)
                    call_id = call.get("id") or f"call_{len(seen_calls)}"
                    check(isinstance(call_id, str) and call_id and call_id not in seen_calls, "Duplicate call id")
                    seen_calls.add(call_id)
                    pending[call_id] = name
                    message["tool_calls"].append({"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": arguments}})
            check(content.strip() or calls, "Empty assistant turn")
        elif role == "tool":
            call_id = original.get("tool_call_id")
            if call_id is None and len(pending) == 1:
                call_id = next(iter(pending))
            check(call_id in pending, "Orphan, ambiguous, or duplicate tool result")
            message["tool_call_id"] = call_id
            message["name"] = pending.pop(call_id)
        output.append(message)
        last_role = role
    check(first_user and last_role == "assistant" and not pending, "Incomplete conversation")
    task = record.get("task_id") or record.get("id") or fingerprint(first_user)
    check(isinstance(task, str) and task, "Invalid task id")
    return {"task_id": task, "prompt_hash": fingerprint(first_user), "messages": output, "tools": tools}


def assistant_spans(rendered):
    """Character ranges for assistant bodies, including their end-of-turn token."""
    marker, end = "<|im_start|>assistant\n", "<|im_end|>"
    spans, cursor = [], 0
    while (start := rendered.find(marker, cursor)) >= 0:
        start += len(marker)
        stop = rendered.find(end, start)
        check(stop >= 0, "Unterminated assistant turn in chat template")
        cursor = stop + len(end)
        spans.append((start, cursor))
    check(spans, "Chat template produced no assistant targets")
    return spans


def encode_record(tokenizer, record, max_length):
    rendered = tokenizer.apply_chat_template(record["messages"], tools=record["tools"] or None,
        tokenize=False, add_generation_prompt=False, enable_thinking=False)
    spans = assistant_spans(rendered)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    check(len(ids) <= max_length, f"Example exceeds max_length ({len(ids)} > {max_length}); refusing truncation")
    labels = [token if any(a <= left < right <= b for a, b in spans) else -100
              for token, (left, right) in zip(ids, offsets)]
    check(any(x != -100 for x in labels), "Example has no supervised assistant tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def parse_completion(text, tools):
    """Parse Qwen3.5 XML calls strictly; return OpenAI-compatible messages.

    JSON-style Hermes calls are also accepted. Never execute generated arguments.
    A malformed call fails the request instead of becoming a plausible chat answer.
    """
    schemas = tool_map(tools)
    # Thinking is internal to this model; never expose it as user-facing content.
    text = text.strip()
    if text.startswith("<think>"):
        check("</think>" in text, "Generation ended inside a thinking block")
        text = text.split("</think>", 1)[1].strip()
    elif text.startswith("</think>"):
        text = text.removeprefix("</think>").strip()
    text = re.sub(r"(?:\s*<\|(?:im_end|endoftext)\|>)+\s*$", "", text).rstrip()
    # Valid Python is literal coding output, including triple-quoted XML examples.
    try:
        ast.parse(text)
        return {'role': 'assistant', 'content': text or None}
    except (SyntaxError, ValueError):
        pass
    protected, fence_start, fence_char, fence_size = [], None, None, 0
    for line in re.finditer(r'.*(?:\n|$)', text):
        marker = re.match(r'^\s{0,3}(`{3,}|~{3,})', line.group())
        if not marker:
            continue
        chars = marker.group(1)
        if fence_start is None:
            fence_start, fence_char, fence_size = line.start(), chars[0], len(chars)
        elif chars[0] == fence_char and len(chars) >= fence_size:
            protected.append((fence_start, line.end()))
            fence_start = None
    if fence_start is not None:
        protected.append((fence_start, len(text)))
    def literal(position):
        return any(a <= position < b for a, b in protected)
    calls, position, contents, call_spans = [], 0, [], []
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, flags=re.S):
        if literal(match.start()):
            continue
        prefix = text[text.rfind('\n', 0, match.start()) + 1:match.start()]
        if prefix.strip() and match.start() != position:
            continue  # Quoted/inline examples are content, not native call blocks.
        contents.append(text[position:match.start()])
        body = match.group(1).strip()
        if body.startswith("{"):
            value = loads(body)
            name, arguments = value.get("name"), value.get("arguments")
            if isinstance(arguments, str):
                arguments = loads(arguments)
        else:
            function = re.fullmatch(r"<function=([a-zA-Z_][a-zA-Z0-9_.-]*)>\s*(.*?)\s*</function>", body, re.S)
            check(function is not None, "Malformed function block")
            name, parameters = function.groups()
            check(name in schemas, f"Unknown tool: {name}")
            arguments, cursor = {}, 0
            properties = schemas[name].get("properties", {})
            for parameter in re.finditer(r"<parameter=([a-zA-Z_][a-zA-Z0-9_.-]*)>\n?(.*?)\n?</parameter>", parameters, re.S):
                check(not parameters[cursor:parameter.start()].strip(), "Malformed parameter block")
                key, value = parameter.groups()
                check(key not in arguments, f"Duplicate tool parameter: {key}")
                arguments[key] = decode_xml_value(value, properties.get(key, {}), schemas[name])
                cursor = parameter.end()
            check(not parameters[cursor:].strip(), "Malformed or unparsed tool parameters")
        validate_call(name, arguments, schemas)
        calls.append({"id": f"call_{uuid.uuid4().hex[:16]}", "type": "function",
                      "function": {"name": name, "arguments": canonical(arguments)}})
        call_spans.append((match.start(), match.end()))
        position = match.end()
    contents.append(text[position:])
    content = "".join(contents).strip()
    for marker in re.finditer(r'(?m)^\s*</?(?:tool_call|function|parameter)(?:>|=)', text):
        if not literal(marker.start() + len(marker.group()) - 1):
            check(any(a <= marker.end() - 1 < b for a, b in call_spans), 'Truncated or malformed tool call')
    message = {"role": "assistant", "content": content or None}
    if calls:
        message["tool_calls"] = calls
    return message

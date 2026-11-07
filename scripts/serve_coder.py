#!/usr/bin/env python3
"""Small OpenAI Chat Completions endpoint for measured local Qwen/LoRA trials.

Loopback only, one inference at a time, bounded bodies/context/output. Streaming
uses a single completed delta (valid SSE, with no token-latency claim).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import time
import uuid

from coder_common import adapter_identity, allocate_device, check, digest, fingerprint, load_coder_model, load_config, loads, parse_completion, runtime_identity, tool_map, verify_model_proof


def chat_stop_tokens(tokenizer, generation_config):
    # Qwen3.5's checkpoint EOS is <|endoftext|>, while assistant turns end
    # with the tokenizer's <|im_end|>. Both must stop generation.
    model_eos = generation_config.eos_token_id
    values = ([model_eos] if isinstance(model_eos, int) else (model_eos or [])) + [tokenizer.eos_token_id]
    check(all(type(value) is int and value >= 0 for value in values), "Invalid chat stop tokens")
    return sorted(set(values))


def validate_request(request, model_name, max_context, max_output):
    check(isinstance(request, dict), "Expected a request object")
    check(request.get("model") == model_name, "Unknown model")
    messages = request.get("messages")
    # The HTTP body and exact rendered-token checks bound resource use. A fixed
    # 200-message cap rejected short tool conversations far below a 262K window.
    check(isinstance(messages, list) and 1 <= len(messages) <= max_context, "Invalid message count")
    for message in messages:
        check(isinstance(message, dict) and message.get("role") in ("system", "user", "assistant", "tool"), "Invalid message")
        content = message.get("content")
        if isinstance(content, list):
            check(all(isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
                      for part in content), "Only text content parts are supported")
            message["content"] = "\n".join(part["text"] for part in content)
        else:
            check(content is None or isinstance(content, str), "Text messages only")
        # OpenAI clients send JSON arguments as strings; the Qwen template needs objects.
        for call in message.get("tool_calls", []):
            function = call["function"]
            if isinstance(function.get("arguments"), str):
                function["arguments"] = loads(function["arguments"])
            check(isinstance(function.get("arguments"), dict), "Tool arguments must be objects")
    tools = request.get("tools") or []
    tool_map(tools)
    choice = request.get("tool_choice", "auto")
    check(choice in ("auto", "none"), "Only auto/none tool_choice is supported")
    if choice == "none":
        tools = []
    # Hermes one-shot may omit both fields. Use the configured/advertised
    # allowance rather than silently truncating a tool patch at 1,024 tokens.
    tokens = request.get("max_completion_tokens", request.get("max_tokens", max_output))
    check(type(tokens) is int and 1 <= tokens <= max_output and tokens < max_context, "Invalid output token limit")
    temperature = request.get("temperature", 0.0)
    check(type(temperature) in (int, float) and math.isfinite(temperature) and 0 <= temperature <= 2, "Invalid temperature")
    check(type(request.get("stream", False)) is bool, "stream must be a boolean")
    stream_options = request.get("stream_options") or {}
    check(isinstance(stream_options, dict) and type(stream_options.get("include_usage", False)) is bool,
          "Invalid stream_options")
    return messages, tools, tokens, temperature


class Backend:
    def __init__(self, cfg, adapter, max_context, max_output, model_path=None, four_bit=False, model_proof=None):
        import torch
        from transformers import AutoTokenizer
        check(torch.cuda.device_count() == 1, "Exactly one allocated CUDA GPU is required")
        self.torch, self.cfg = torch, cfg
        self.max_context, self.max_output = max_context, max_output
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path or cfg["model_id"]), revision=cfg["revision"],
            trust_remote_code=False, local_files_only=bool(model_path))
        self.model = load_coder_model(cfg, model_path, four_bit)
        self.identity = {"identity_schema": 2, "base_model": cfg["model_id"], "revision": cfg["revision"], "adapter": None,
                         "model_proof_sha256": fingerprint(model_proof) if model_proof else None,
                         "limits": {"context": max_context, "output": max_output},
                         "quantization": "nf4-double" if four_bit else "bf16", "runtime": runtime_identity(),
                         "protocol_sha256": fingerprint({p.name: digest(p) for p in
                             (Path(__file__), Path(__file__).with_name("coder_common.py"))})}
        if adapter:
            from peft import PeftModel
            result_path = adapter.parent / "training-result.json"
            check(result_path.is_file(), "Adapter is missing training-result.json")
            result = loads(result_path.read_text())
            run = loads((adapter.parent / "run-manifest.json").read_text())
            completed = loads((adapter.parent / "completed-run-manifest.json").read_text())
            check(completed["status"] == "complete" and completed["exit_code"] == 0 and
                  result["status"] == "trained_unpromoted", "Adapter run is incomplete")
            check(run["model_id"] == cfg["model_id"] and run["model_revision"] == cfg["revision"],
                  "Adapter was trained against a different base model")
            check(not result["smoke_only"], "A smoke-test adapter cannot be served as the candidate")
            for name, expected in result["adapter_files"].items():
                check(Path(name).name == name and digest(adapter / name) == expected, "Adapter checksum mismatch")
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=False)
            self.identity["adapter"] = str(adapter)
            self.identity.update(adapter_identity(result))
        self.model.eval()
        self.identity["chat_stop_token_ids"] = chat_stop_tokens(self.tokenizer, self.model.generation_config)

    def complete(self, request):
        name = self.cfg["served_model_name"]
        messages, tools, limit, temperature = validate_request(request, name, self.max_context, self.max_output)
        rendered = self.tokenizer.apply_chat_template(messages, tools=tools or None,
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        count = inputs.input_ids.shape[1]
        check(count + limit <= self.max_context, "Context limit exceeded; compress the conversation")
        options = {"max_new_tokens": limit, "do_sample": temperature > 0, "use_cache": True,
                   "pad_token_id": self.tokenizer.eos_token_id,
                   "eos_token_id": chat_stop_tokens(self.tokenizer, self.model.generation_config)}
        if temperature > 0:
            options.update(temperature=temperature, top_p=0.95)
        start = time.monotonic()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **options)[0, count:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        message = parse_completion(raw, tools)
        finish = "tool_calls" if message.get("tool_calls") else ("length" if len(generated) >= limit else "stop")
        return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
                "model": name, "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": count, "completion_tokens": len(generated), "total_tokens": count + len(generated)},
                "wsp": {"identity": self.identity, "inference_seconds": time.monotonic() - start}}


def handler_for(backend):
    instance = uuid.uuid4().hex
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *_args):
            pass

        def send_json(self, code, value):
            blob = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def do_GET(self):
            if self.path in ("/health", "/v1/models"):
                self.send_json(200, {"object": "list", "data": [{"id": backend.cfg["served_model_name"],
                    "object": "model", "owned_by": "local"}], "wsp": backend.identity, "wsp_instance": instance})
            else:
                self.send_json(404, {"error": {"message": "Unknown route"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": {"message": "Unknown route"}})
                return
            try:
                self.connection.settimeout(30)
                size = int(self.headers.get("Content-Length", "0"))
                check(0 < size <= 2 * 1024 * 1024, "Invalid request body size")
                raw = self.rfile.read(size)
                check(len(raw) == size, "Incomplete request body")
                request = loads(raw)
                result = backend.complete(request)
                result['wsp'] = {**result.get('wsp', {}), 'identity': backend.identity, 'instance_id': instance}
            except (ValueError, KeyError, TypeError) as error:
                self.close_connection = True
                print(f'Request/completion validation failed: {type(error).__name__}: {error}', flush=True)
                self.send_json(400, {"error": {"message": str(error), "type": "invalid_request_error"}})
                return
            except Exception as error:
                self.close_connection = True
                print(f"Inference failed: {type(error).__name__}: {error}", flush=True)
                self.send_json(500, {"error": {"message": "Inference failed; see the local server log"}})
                return
            if not request.get("stream"):
                self.send_json(200, result)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            message = result["choices"][0]["message"]
            delta = {"role": "assistant", "content": message["content"]}
            if message.get("tool_calls"):
                delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]
            common = {k: result[k] for k in ("id", "created", "model")}
            for change, finish in ((delta, None), ({}, result["choices"][0]["finish_reason"])):
                chunk = {**common, "wsp": result['wsp'], "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": change, "finish_reason": finish}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            if (request.get("stream_options") or {}).get("include_usage"):
                chunk = {**common, "wsp": result['wsp'], "object": "chat.completion.chunk", "choices": [], "usage": result["usage"]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/qwen35-4b.json")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--max-output", type=int, default=2048)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-proof", type=Path, required=True)
    parser.add_argument("--four-bit", action="store_true")
    parser.add_argument("--direct-idle-gpu", action="store_true")
    args = parser.parse_args()
    check(1 <= args.port <= 65535 and 512 <= args.max_context <= 32768 and 1 <= args.max_output < args.max_context, "Invalid server limits")
    cfg = load_config(args.config)
    proof = verify_model_proof(cfg, args.model_path, args.model_proof)
    allocate_device(args.direct_idle_gpu)
    backend = Backend(cfg, args.adapter, args.max_context, args.max_output, args.model_path, args.four_bit, proof)
    server = HTTPServer(("127.0.0.1", args.port), handler_for(backend))
    print(json.dumps({"status": "ready", "url": f"http://127.0.0.1:{args.port}/v1", "identity": backend.identity,
                      "created_at": datetime.now(timezone.utc).isoformat()}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

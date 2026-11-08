#!/usr/bin/env python3
"""Pinned BF16 LoRA adaptation of Qwen3.5-4B's language layers on one queued GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from coder_common import allocate_device, check, digest, encode_record, load_coder_model, load_config, runtime_identity, verify_model_proof, write_json
from prepare_coder_data import read_dataset
from coder_artifacts import capture_recipe
from coder_initialization import initialize_adapter, verify_initial_adapter


def adapter_targets(named_modules, linear_type, scope='all-language'):
    """Select an explicit adaptation scope while keeping vision/head frozen."""
    check(isinstance(scope, str) and scope in ('all-language', 'mlp'), 'Invalid LoRA target scope')
    targets = [name for name, module in named_modules
               if '.language_model.layers.' in name and isinstance(module, linear_type)
               and (scope == 'all-language' or '.mlp.' in name)]
    check(targets, 'No language-layer linear modules found for the requested scope')
    return targets


def main():
    from method_policy import blocked_legacy_operation
    blocked_legacy_operation('LoRA/QLoRA training')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/qwen35-4b.json")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value explicitly marks a smoke experiment")
    parser.add_argument("--model-path", type=Path, required=True, help="Exact authenticated pinned snapshot")
    parser.add_argument("--model-proof", type=Path, required=True)
    parser.add_argument("--four-bit", action="store_true", help="NF4 QLoRA for the 12 GB .54 GPU")
    parser.add_argument("--direct-idle-gpu", action="store_true", help="Explicit admission on .54, which has no GPUQ")
    parser.add_argument("--init-training", type=Path, help="Authenticated completed adapter; starts a new optimizer, not an optimizer resume")
    args = parser.parse_args()
    cfg = load_config(args.config)
    verify_model_proof(cfg, args.model_path, args.model_proof)
    manifest, splits = read_dataset(args.data)
    initial = None
    if args.init_training is not None:
        check(not args.output.resolve().is_relative_to(args.init_training.resolve())
              and not args.init_training.resolve().is_relative_to(args.output.resolve()),
              "Training output overlaps the immutable initial adapter")
        initial = verify_initial_adapter(args.init_training, cfg)
    allocation = allocate_device(args.direct_idle_gpu)
    check(args.max_steps == -1 or args.max_steps > 0, "Invalid max_steps")
    check(not args.output.exists() and not args.output.is_symlink(), "Refusing existing training output")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

    check(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one allocated CUDA GPU is required")
    set_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path or cfg["model_id"]),
        revision=cfg["revision"], trust_remote_code=False, local_files_only=bool(args.model_path))
    check(tokenizer.is_fast, "Assistant loss masking requires offset mappings from a fast tokenizer")
    encoded, dropped = {}, {}
    for name, rows in splits.items():
        encoded[name], dropped[name] = [], []
        for row in rows:
            try:
                encoded[name].append(encode_record(tokenizer, row, cfg["max_length"]))
            except ValueError as error:
                if "exceeds max_length" not in str(error):
                    raise
                dropped[name].append(row["task_id"])
        minimum = cfg["minimum_train_examples" if name == "train" else "minimum_validation_examples"]
        check(len(encoded[name]) >= minimum, f"Too few usable {name} examples: {len(encoded[name])} < {minimum}")

    args.output.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    patch = subprocess.check_output(["git", "diff", "HEAD"], cwd=Path(__file__).resolve().parents[1])
    # Hash every executed recipe file too: git diff does not include new untracked files.
    recipe = capture_recipe(Path(__file__).resolve().parents[1], args.output, args.config)
    run = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
           "host": os.uname().nodename, "source_revision": source,
           "dirty_patch_sha256": __import__("hashlib").sha256(patch).hexdigest(), "recipe_sha256": recipe,
           "model_id": cfg["model_id"], "model_revision": cfg["revision"], "config": cfg,
           "config_sha256": digest(args.config), "dataset_manifest_sha256": digest(args.data / "dataset.json"),
           "sources": manifest["sources"], "command_argv": sys.argv, "requested_gpu_count": 1,
           "allocated_devices": os.environ["CUDA_VISIBLE_DEVICES"],
           "allocation_mode": allocation, "quantization": "nf4-double" if args.four_bit else "bf16",
           "gpu_allocator_config": os.environ.get("PYTORCH_ALLOC_CONF"),
           "model_path": str(args.model_path) if args.model_path else None,
           "model_proof_sha256": digest(args.model_proof),
           "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "accelerate")},
           "runtime": runtime_identity(),
           "examples": {n: len(v) for n, v in encoded.items()}, "oversize_tasks_dropped": dropped,
           "supervised_tokens": {n: sum(sum(t != -100 for t in r["labels"]) for r in v) for n, v in encoded.items()},
           "initial_adapter": initial, "smoke_only": args.max_steps > 0, "status": "running"}
    write_json(args.output / "run-manifest.json", run)
    model = load_coder_model(cfg, args.model_path, args.four_bit)
    if args.four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    targets = adapter_targets(model.named_modules(), torch.nn.Linear, cfg.get('lora_target_scope', 'all-language'))
    model = get_peft_model(model, LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        target_modules=targets, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    if args.init_training is not None:
        write_json(args.output / "initial-adapter.json", initialize_adapter(model, args.init_training, cfg, targets))
    model.print_trainable_parameters()
    write_json(args.output / "adapter-targets.json", targets)

    def collate(rows):
        size = max(len(row["input_ids"]) for row in rows)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        result = {}
        for key, value in (("input_ids", pad), ("attention_mask", 0), ("labels", -100)):
            result[key] = torch.tensor([row[key] + [value] * (size - len(row[key])) for row in rows], dtype=torch.long)
        return result

    # Admit the longest training example with the actual backward/optimizer path
    # before spending time on a full epoch. The probe optimizer has zero learning
    # rate and weight decay, so this is a memory/finite-gradient check only.
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    largest = max(encoded["train"], key=lambda row: len(row["input_ids"]))
    probe_batch = {key: value.to(model.device) for key, value in collate([largest]).items()}
    probe_optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0, weight_decay=0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        probe_loss = model(**probe_batch).loss
    check(torch.isfinite(probe_loss).item(), "Non-finite worst-case probe loss")
    probe_loss.backward()
    check(all(torch.isfinite(parameter.grad).all().item() for parameter in model.parameters() if parameter.grad is not None),
          "Non-finite worst-case probe gradients")
    probe_optimizer.step()
    model.zero_grad(set_to_none=True)
    write_json(args.output / "memory-probe.json", {"input_tokens": len(largest["input_ids"]),
        "loss": probe_loss.item(), "learning_rate": 0, "finite_gradients": True,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()})
    del probe_batch, probe_optimizer, probe_loss
    __import__("gc").collect()
    torch.cuda.empty_cache()
    set_seed(cfg["seed"])
    planned_steps = args.max_steps if args.max_steps > 0 else math.ceil(
        math.ceil(len(encoded["train"]) / cfg["gradient_accumulation_steps"]) * cfg["epochs"])
    checkpoint_steps = max(1, math.ceil(planned_steps / 4))
    training = TrainingArguments(output_dir=str(args.output / "checkpoints"),
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"], learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"], max_steps=args.max_steps, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", lr_scheduler_type="cosine", warmup_steps=0.05,
        logging_steps=5, save_strategy="steps", save_steps=checkpoint_steps,
        eval_strategy="steps", eval_steps=checkpoint_steps, save_total_limit=4,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        max_grad_norm=cfg["max_grad_norm"], report_to="none", seed=cfg["seed"],
        remove_unused_columns=False, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=training, train_dataset=encoded["train"],
                      eval_dataset=encoded["validation"], data_collator=collate)
    baseline = trainer.evaluate()
    check(math.isfinite(baseline["eval_loss"]), "Non-finite baseline validation loss")
    write_json(args.output / "validation-before.json", baseline)
    trained = trainer.train()
    after = trainer.evaluate()
    check(math.isfinite(after["eval_loss"]) and math.isfinite(trained.metrics["train_loss"]), "Non-finite training metrics")
    adapter = args.output / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    metrics = {"status": "trained_unpromoted", "smoke_only": args.max_steps > 0,
               "before": baseline, "after": after, "training": trained.metrics,
               "selected_checkpoint": trainer.state.best_model_checkpoint,
               "selected_validation_loss": trainer.state.best_metric,
               "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
               "gpu_name": torch.cuda.get_device_name(0),
               "adapter_files": {p.name: digest(p) for p in sorted(adapter.iterdir()) if p.is_file()},
               "acceptance": "Requires independent coding, multi-turn tool, and real client evaluation"}
    write_json(args.output / "training-result.json", metrics)
    write_json(args.output / "completed-run-manifest.json", {**run, "status": "complete", "exit_code": 0,
        "validation": {"finite_losses": True, "adapter_saved": True, "promoted": False},
        "expected_outputs": ["adapter/adapter_model.safetensors", "adapter/adapter_config.json", "training-result.json"],
        "job_id": os.environ.get("INVOCATION_ID"), "gpu_name": torch.cuda.get_device_name(0)})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

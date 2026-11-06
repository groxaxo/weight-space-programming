#!/usr/bin/env python3
"""Export an earlier saved checkpoint from a complete run as an unpromoted candidate.

Validation loss is taken from that checkpoint's actual Trainer state. Original
training artifacts remain untouched. This operation performs no GPU training.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

from coder_common import check, digest, loads, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.training.resolve()
    completed = loads((source / "completed-run-manifest.json").read_text())
    result = loads((source / "training-result.json").read_text())
    check(completed["status"] == "complete" and completed["exit_code"] == 0 and
          result["smoke_only"] is False, "Need a complete full training run")
    for name, expected in result["adapter_files"].items():
        check(Path(name).name == name and digest(source / "adapter" / name) == expected,
              "Source adapter checksum mismatch")
    check(args.step > 0, "Checkpoint step must be positive")
    checkpoint = source / "checkpoints" / f"checkpoint-{args.step}"
    state = loads((checkpoint / "trainer_state.json").read_text())
    check(state["global_step"] == args.step, "Checkpoint step does not match Trainer state")
    evaluations = [row for row in state["log_history"] if row.get("step") == args.step and "eval_loss" in row]
    check(len(evaluations) == 1, "Need the checkpoint's own validation result")
    after = evaluations[0]
    check(after["eval_loss"] >= 0, "Invalid checkpoint validation loss")
    cfg = loads((checkpoint / "adapter_config.json").read_text())
    final_cfg = loads((source / "adapter/adapter_config.json").read_text())
    for name in ("base_model_name_or_path", "r", "lora_alpha", "target_modules", "peft_type"):
        check(cfg[name] == final_cfg[name], "Checkpoint adapter architecture differs from the completed run")

    from safetensors import safe_open
    import torch
    torch.set_num_threads(2)
    count, parameters = 0, 0
    targets = loads((source / "adapter-targets.json").read_text())
    expected_names = {f"base_model.model.{target}.lora_{part}.weight" for target in targets for part in ("A", "B")}
    with safe_open(checkpoint / "adapter_model.safetensors", framework="pt", device="cpu") as tensors:
        check(set(tensors.keys()) == expected_names, "Incomplete or unexpected checkpoint adapter tensors")
        for name in tensors.keys():
            check(re.search(r"\.language_model\.layers\.\d+\..*\.lora_[AB]\.weight$", name),
                  "Unexpected non-language adapter tensor")
            tensor = tensors.get_tensor(name)
            check(torch.isfinite(tensor).all().item(), "Non-finite checkpoint adapter tensor")
            count += 1
            parameters += tensor.numel()
    check(count == len(targets) * 2, "Incomplete checkpoint adapter")
    check(not args.output.exists() and not args.output.is_symlink(), "Refusing existing checkpoint export")
    args.output.mkdir(parents=True, exist_ok=False)
    adapter = args.output / "adapter"
    adapter.mkdir()
    for name in result["adapter_files"]:
        origin = checkpoint / name if name in ("adapter_config.json", "adapter_model.safetensors", "training_args.bin") else source / "adapter" / name
        check(origin.is_file() and not origin.is_symlink(), "Missing regular adapter artifact")
        shutil.copyfile(origin, adapter / name)
    derivation = {"operation": "saved_checkpoint_selection", "step": args.step,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(checkpoint),
        "source_training_result_sha256": digest(source / "training-result.json"),
        "source_completed_manifest_sha256": digest(source / "completed-run-manifest.json"),
        "trainer_state_sha256": digest(checkpoint / "trainer_state.json"),
        "selection_script_sha256": digest(Path(__file__)),
        "reason": "Earlier checkpoint candidate for development and independent benchmark/client evaluation"}
    run = loads((source / "run-manifest.json").read_text())
    write_json(args.output / "run-manifest.json", {**run, "checkpoint_selection": derivation})
    shutil.copytree(source / "source", args.output / "source")
    shutil.copyfile(Path(__file__), args.output / "source/select_coder_checkpoint.py")
    shutil.copyfile(source / "adapter-targets.json", args.output / "adapter-targets.json")
    metrics = {"status": "trained_unpromoted", "smoke_only": False, "before": result["before"], "after": after,
        "selected_checkpoint": str(checkpoint), "selected_validation_loss": after["eval_loss"],
        "training": {"optimizer_steps": args.step, "epoch": after["epoch"], "source_run": str(source)},
        "derivation": derivation, "adapter_files": {p.name: digest(p) for p in sorted(adapter.iterdir())},
        "acceptance": "Requires fresh paired development, HumanEval and real-client results"}
    write_json(args.output / "adapter-verification.json", {"status": "verified", "finite_tensors": count,
        "trainable_parameters": parameters, "language_layers_only": True})
    write_json(args.output / "completed-run-manifest.json", {**completed, "checkpoint_selection": derivation,
        "parent_job_id": completed["job_id"], "job_id": None})
    write_json(args.output / "training-result.json", metrics)
    print(json.dumps({"output": str(args.output), "step": args.step, "validation_loss": after["eval_loss"],
                      "adapter_sha256": metrics["adapter_files"]["adapter_model.safetensors"]}))


if __name__ == "__main__":
    main()

# Weight-Space Programming Lab

Reproducible, non-destructive reproduction of the published Qwen3.5-0.8B 50-pass edit, with upstream OpenWSP checked out for future independent editing experiments.

## Status
- Upstream research and OpenWSP cloned.
- Published edit and its manifest available locally.
- Pipeline preflight and CPU-only reconstruction wrapper included.
- **Full checkpoint rebuild and GPU benchmarks are not yet verified.**
- Do not use the benchmark datasets to select edits; keep training/practice and held-out evaluation separate.

## Commands
```bash
cd /home/op/experiments/weight-space-programming
python3 scripts/pipeline.py preflight
python3 -m venv .venv
.venv/bin/pip install numpy huggingface_hub
.venv/bin/python scripts/pipeline.py download
.venv/bin/python scripts/pipeline.py rebuild
```

The upstream reconstruction checks SHA-256 of every modified tensor against the authors' evaluated checkpoint and refuses incompatible base weights. This wrapper pins the original revision, prevents overwriting the base and writes a provenance record to reports/rebuild.json.

**Evaluation:** See upstream/lm_understand-logs/exp8_math/code/evaluate.py and profiles.py for the exact authors' vLLM protocol. Run original-thinking, edited-thinking, and original-non-thinking under matching decoding budgets. Do not interpret a short smoke test as reproduction of published MATH-500 scores.

**Future discovery:** upstream/openwsp is a separate research implementation, not automatically integrated into the reconstruction wrapper. Review its environment and isolation requirements before executing agent-generated edits. Preserve immutable baselines and compare against held-out benchmarks.

GPU usage is intentionally opt-in; no existing services are stopped.

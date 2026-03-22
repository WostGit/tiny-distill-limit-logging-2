# tiny-distill-limit-logging-2

This repository is a **self-contained, GitHub Actions–first smoke test** for tiny LoRA-style distillation on CPU runners. It intentionally uses a tiny checked-in dataset, a tiny model, and one-epoch training so the run finishes in CI while producing very verbose logs that expose practical scaling limits (time, RAM, token length, checkpoint I/O, evaluation overhead, preprocessing overhead, and thread behavior).

> This is a **limit-diagnosis smoke test repo**, not a full distillation benchmark framework.

## Why this repo is self-contained

Everything required to run is checked in:

- tiny dataset in `data/`
- training + evaluation scripts in `scripts/`
- logging + metrics helpers in `scripts/`
- CI workflow in `.github/workflows/`
- dependencies in `requirements.txt`
- output metrics path in `outputs/metrics/`

No monorepo helpers, shared internal packages, or hidden project glue are required.

## Quick local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_tiny_distill.py
python scripts/eval_tiny_distill.py
```

Metrics JSON files are written to:

- `outputs/metrics/train_metrics.json`
- `outputs/metrics/eval_metrics.json`

## GitHub Actions smoke test

Workflow file: `.github/workflows/smoke-test.yml`

The workflow runs on `ubuntu-latest`, installs `requirements.txt`, executes training and eval, and uploads `outputs/metrics/*.json` as an artifact.

## How to read the verbose limit logs

Look for these log blocks and tags:

1. **Environment instrumentation**
   - Python version, torch version, OS/platform
   - Torch thread settings (`torch_num_threads`, `torch_num_interop_threads`)

2. **Data pipeline instrumentation**
   - `[DATA] Loaded ...` (data load time)
   - `[DATA] Tokenized/preprocessed ...` (preprocessing time)
   - `[LEN] ...` (input/target min/mean/max/p95 sequence lengths)

3. **Training step instrumentation**
   - `[STEP] opt_step=... elapsed=... step_duration=...`
   - includes cumulative samples and token counters

4. **Memory snapshots**
   - `[MEM] tag=process_start`
   - `[MEM] tag=after_model_load`
   - `[MEM] tag=after_lora_wrapping`
   - `[MEM] tag=after_first_forward_backward`
   - `[MEM] tag=optimizer_step_*`
   - `[MEM] tag=before_save`, `after_save`, `before_eval`, `after_eval`

5. **Checkpoint instrumentation**
   - `[CKPT] Save start/end ...`
   - save duration, checkpoint size (MB), and file count

6. **Evaluation instrumentation**
   - adapter reload time
   - eval duration
   - generation duration + tiny preview

Finally, inspect the printed **Final Bottleneck Summary** in train/eval logs and compare with JSON artifacts.

## Expected bottlenecks

On standard GitHub-hosted CPU runners, likely first limits are usually:

- wall-clock time per optimizer step
- preprocessing overhead if tokenizer setup dominates tiny runs
- evaluation overhead if generation is enabled
- checkpoint I/O overhead if repeated saves are added

RAM pressure and token-length effects are still surfaced by memory snapshots and sequence-length stats, even in tiny runs.

## What this smoke test proves and does not prove

### Proves

- the tiny end-to-end distillation path runs in CI
- verbose instrumentation can identify the first practical bottleneck category
- JSON artifacts are emitted for post-run comparison

### Does not prove

- quality competitiveness on real-world tasks
- scaling behavior for larger models/datasets
- final architecture choices for production distillation

Treat this as an early warning diagnostic harness, not as a benchmark leaderboard entry.

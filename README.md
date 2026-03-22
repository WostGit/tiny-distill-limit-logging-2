# tiny-distill-limit-logging-2

`tiny-distill-limit-logging-2` is a **self-contained smoke test repository** that runs a tiny LoRA-based distillation experiment on CPU and emits intentionally verbose logs so you can see where scaling limits appear first on standard GitHub Actions runners. This is **GitHub Actions first**: the included workflow installs dependencies, runs train + eval, and uploads JSON metrics artifacts.

## What is included

- Tiny checked-in datasets (`data/tiny_train.jsonl`, `data/tiny_eval.jsonl`)
- Explicit training script with tiny teacher/student distillation (`scripts/train_tiny_distill.py`)
- Explicit evaluation script that reloads the adapter (`scripts/eval_tiny_distill.py`)
- Logging and metrics helpers (`scripts/logging_utils.py`, `scripts/metrics_utils.py`)
- GitHub Actions smoke workflow (`.github/workflows/smoke-test.yml`)
- Versioned output artifact directory (`outputs/metrics/.gitkeep`)

## GitHub Actions first

The primary entrypoint is `.github/workflows/smoke-test.yml` on `ubuntu-latest` with CPU only. It runs:

1. `python scripts/train_tiny_distill.py`
2. `python scripts/eval_tiny_distill.py`
3. Artifact upload for `outputs/metrics/*.json`

No repository-external private utilities or unpublished packages are required.

## How to read the verbose limit logs

Training and evaluation logs are designed for diagnosis, not brevity. Look for these markers:

- **Per optimizer step timing**: `OPT STEP idx=... step_duration=... cumulative_samples=...`
- **Memory snapshots** at critical points:
  - process start
  - after model load
  - after LoRA wrapping
  - after first forward/backward
  - every optimizer step
  - before/after save
  - before/after eval
- **Sequence stats** in output JSON:
  - input token length min/mean/max/p95
  - target token length min/mean/max/p95
  - tokens per optimizer step
- **Training shape metadata**:
  - batch size
  - grad accumulation steps
  - effective samples/optimizer step
- **Checkpoint IO instrumentation**:
  - save start/end timing
  - directory byte size
  - file count
- **Eval instrumentation**:
  - adapter reload time
  - eval duration
  - generation duration
- **Data pipeline instrumentation**:
  - data load time
  - tokenization/preprocessing time
- **Environment/thread instrumentation**:
  - Python + torch version
  - OS/platform details
  - torch thread counts + thread env vars

## Expected bottlenecks

On small CI runs, the first limit is often one of:

- Wall-clock overhead from Python + model forward/backward on CPU
- Checkpoint IO overhead relative to tiny training duration
- Evaluation overhead becoming dominant when train is very short
- Thread behavior (under/over-utilization) impacting step consistency

Memory and token-length limits are included in the logs, but with this tiny config they are usually informative baselines rather than hard-failure triggers.

## What this smoke test proves and does not prove

### Proves

- The repo can run tiny end-to-end distillation + eval in CI
- You can inspect logs/metrics to identify likely first bottlenecks
- Instrumentation captures timing, memory, data, checkpoint, eval, and environment details

### Does not prove

- SOTA model quality or benchmark performance
- Production-ready training stability at large scale
- Generalizable scaling behavior across hardware, datasets, or model families

This repository is intentionally a **smoke test and limit-diagnosis repo**, not a full distillation benchmark.

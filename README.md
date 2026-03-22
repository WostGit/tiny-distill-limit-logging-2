# tiny-distill-limit-logging-2

This repository is a **self-contained tiny distillation smoke test** designed to run on GitHub Actions and emit intentionally verbose logs that reveal practical scaling limits early. It trains a tiny LoRA-wrapped student against a tiny teacher objective on a checked-in dataset, saves an adapter checkpoint, reloads it, and runs tiny evaluation while logging timing, memory, sequence length, IO, and thread instrumentation.

This repo is **self-contained**: dataset, scripts, workflow, requirements, and instrumentation all live here.

This repo is **GitHub Actions first**: CPU-only, `ubuntu-latest`, deterministic defaults, and metrics artifacts uploaded on each CI run.

> This is a **smoke test and limit-diagnosis repo**, not a full distillation benchmark.

## Repository layout

- `data/tiny_train.jsonl`, `data/tiny_eval.jsonl`: tiny checked-in dataset.
- `scripts/train_tiny_distill.py`: tiny LoRA distillation training + post-train eval call.
- `scripts/eval_tiny_distill.py`: standalone adapter reload and evaluation instrumentation.
- `scripts/logging_utils.py`: verbose event/memory/environment logging helpers.
- `scripts/metrics_utils.py`: compact JSON/JSONL writing and timing summarization.
- `.github/workflows/smoke-test.yml`: end-to-end CI smoke workflow.
- `outputs/metrics/`: JSON metrics artifact target.

## Quick local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_tiny_distill.py
python scripts/eval_tiny_distill.py
```

## How to read the verbose limit logs

The smoke test emits both console logs and JSON metrics files under `outputs/metrics/`.

### Console logs
- `[ENV] ...`: environment snapshot (Python, torch, OS, thread settings).
- `[MEMORY] ...`: RSS/VMS snapshots at important lifecycle stages.
- `[EVENT] ...`: structured events including step timing and checkpoint instrumentation.
- `=== FINAL BOTTLENECK SUMMARY ===`: compact human-oriented bottleneck diagnosis.

### Metrics files
- `outputs/metrics/train_run_metrics.json`: full training run summary.
- `outputs/metrics/train_step_metrics.jsonl`: per-optimizer-step timing and token stats.
- `outputs/metrics/post_train_eval_metrics.json`: eval called from training script.
- `outputs/metrics/eval_metrics.json`: standalone eval script metrics.

## Expected bottlenecks

On standard GitHub-hosted CPU runners, the first limit is commonly one of:
- Wall-clock time in model forward/backward,
- Model load + checkpoint IO overhead,
- Preprocessing/data path overhead for very small runs,
- Eval/generation overhead relative to tiny training,
- Threading inefficiency (too many/few CPU threads),
- RAM pressure if sequence lengths or model size are increased.

## What this smoke test proves and does not prove

### Proves
- The training/eval pipeline works end-to-end in CI.
- Instrumentation can identify likely first bottlenecks.
- Adapter save/reload and metric artifact generation function correctly.

### Does not prove
- SOTA quality or comprehensive distillation performance.
- Transfer to larger models, GPUs, or long-context regimes.
- Production-ready throughput or cost efficiency.

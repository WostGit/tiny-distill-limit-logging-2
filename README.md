# tiny-distill-limit-logging-2

`tiny-distill-limit-logging-2` is a **self-contained tiny distillation smoke test** repository that runs on **GitHub Actions first**, trains a very small LoRA-adapted student model on a checked-in toy dataset, and emits intentionally verbose logs plus compact JSON metrics so you can quickly see what the first scaling limit is (time, RAM, token length, checkpoint IO, eval overhead, preprocessing overhead, or thread behavior).

This repo includes everything needed in one place: tiny train/eval datasets, training script, evaluation script, logging utilities, metrics utilities, dependencies, CI workflow, and contributor guidance.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_tiny_distill.py
python scripts/eval_tiny_distill.py --adapter-dir outputs/checkpoints/tiny_lora_adapter
```

## GitHub Actions first

The canonical run path is `.github/workflows/smoke-test.yml` on `ubuntu-latest` CPU runners.
The workflow:
1. installs dependencies from `requirements.txt`
2. runs `scripts/train_tiny_distill.py`
3. runs `scripts/eval_tiny_distill.py`
4. uploads `outputs/metrics/*.json` as artifacts

## How to read the verbose limit logs

The training and evaluation scripts emit structured JSON lines to stdout. Key event groups:

- **Per-optimizer-step timing**: elapsed-from-start, step duration, cumulative samples, and optimizer step index.
- **Memory snapshots**: process start, post-model-load, post-LoRA-wrap, post-first-forward/backward, each optimizer step, pre-save/post-save, pre-eval/post-eval.
- **Sequence stats**: input and target token length min/mean/max/p95, plus tokens processed per optimizer step.
- **Training-shape metadata**: batch size, grad accumulation, effective samples per optimizer step.
- **Checkpoint instrumentation**: save start/end, save duration, checkpoint file count and bytes.
- **Eval instrumentation**: adapter reload time, eval duration, generation duration.
- **Data pipeline instrumentation**: preprocessing/tokenization time and dataloader setup time.
- **Environment instrumentation**: python/torch/platform/thread settings.

Use the final `final_bottleneck_summary` event from training as the first-pass diagnosis, then verify by checking the raw step-level and phase-level logs.

## Expected bottlenecks

Because this is intentionally tiny and CPU-only, likely first limits are:

- fixed overheads (model load + preprocessing) dominating total runtime
- checkpoint IO as a noticeable fraction of wall-clock
- eval/generation overhead competing with training time
- CPU thread configuration creating throughput variance

For this toy scale, GPU-memory-style bottlenecks are less likely than orchestration/overhead bottlenecks.

## What this smoke test proves and does not prove

### Proves

- The end-to-end tiny LoRA-style distillation path works in CI.
- Instrumentation is sufficient to identify the *first* bottleneck category in a small run.
- JSON metrics artifacts are produced for post-run analysis.

### Does **not** prove

- Absolute model quality on real tasks.
- General scaling behavior on larger models/datasets/hardware.
- Production-ready distillation performance tuning.

This repository is intentionally a **smoke test and limit-diagnosis repo**, not a full distillation benchmark suite.

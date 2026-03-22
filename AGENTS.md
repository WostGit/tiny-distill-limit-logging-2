# Agent Guidance for tiny-distill-limit-logging-2

This repository is a focused, self-contained experiment for diagnosing scaling limits of tiny LoRA-based distillation on GitHub Actions CPU runners.

## Priorities

1. Preserve verbose, limit-focused instrumentation.
2. Keep the repo fully self-contained.
3. Prefer small, reviewable changes.
4. Keep GitHub Actions compatibility as a top priority.

## Logging constraints

- Do **not** replace explicit logs with opaque abstractions that hide timing or memory details.
- Do **not** remove useful timing/memory/checkpoint/eval logs unless replaced with something better and equally explicit.
- Keep per-step and per-phase diagnostics visible in normal CI logs.

## Scope constraints

- This is a smoke-test + limit-diagnosis repo, not a general training framework.
- Avoid introducing unnecessary architectural layers.
- Keep dependencies minimal and easy to install on `ubuntu-latest`.

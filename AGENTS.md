# AGENTS Instructions

This repository is a focused tiny distillation smoke-test project for CI limit diagnosis.

## Core expectations

- Preserve verbose, limit-focused instrumentation.
- Keep the repository self-contained.
- Prefer small, reviewable changes.
- Do not replace explicit logging with opaque abstractions.
- Do not remove useful timing or memory logs without a clearly better replacement.
- Keep GitHub Actions compatibility as a top priority.

## Implementation style

- Favor explicit, readable Python over helper-heavy abstractions.
- Keep train/eval behavior deterministic where practical.
- Keep CPU-only defaults and runner-friendly settings.
- Ensure logs remain useful in GitHub Actions output.

## Artifacts

- Keep JSON metrics emission working.
- Keep `outputs/metrics/` committed with `.gitkeep` so artifact paths always exist.

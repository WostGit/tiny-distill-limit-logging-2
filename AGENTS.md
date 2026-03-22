# AGENTS.md

## Purpose
This repository is a tiny, self-contained LoRA distillation smoke test focused on verbose limit-diagnosis logs for GitHub Actions CPU runners.

## Agent instructions
- Preserve verbose, limit-focused instrumentation.
- Keep the repository self-contained and easy to understand from a fresh clone.
- Prefer small, reviewable changes over broad refactors.
- Do not replace explicit logging with opaque abstractions.
- Do not remove useful timing or memory logs without a clearly better replacement.
- Keep GitHub Actions compatibility as a top priority.
- Maintain compact JSON metrics artifacts under `outputs/metrics/`.

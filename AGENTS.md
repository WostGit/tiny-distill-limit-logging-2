# AGENTS instructions for tiny-distill-limit-logging-2

This repository is a **self-contained smoke-test** for diagnosing tiny LoRA distillation limits on GitHub Actions.

## Priorities
1. Preserve and improve verbose limit-focused instrumentation.
2. Keep the repo fully self-contained (no hidden dependencies, no external local project assumptions).
3. Prefer small, reviewable changes over broad refactors.
4. Keep explicit logging readable; do not replace with opaque abstractions.
5. Do not remove useful timing or memory logs unless replaced with better diagnostics.
6. Maintain first-class GitHub Actions compatibility on `ubuntu-latest` CPU runners.

## Coding expectations
- Make bottleneck causes visible in logs (time, RAM, token length, checkpoint IO, eval overhead, preprocessing, thread behavior).
- Keep scripts straightforward and easy for reviewers to inspect quickly.
- Emit compact JSON metrics artifacts alongside human-readable console logs.

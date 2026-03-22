import json
import os
from pathlib import Path
from typing import Any, Dict


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_metrics_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[METRICS] Wrote JSON metrics to {path}")


def directory_size_and_count(path: str) -> Dict[str, float]:
    total_bytes = 0
    file_count = 0
    for root, _, files in os.walk(path):
        for fname in files:
            file_count += 1
            full_path = os.path.join(root, fname)
            total_bytes += os.path.getsize(full_path)
    return {
        "bytes": float(total_bytes),
        "megabytes": float(total_bytes / (1024 ** 2)),
        "file_count": float(file_count),
    }

import json
import math
import os
import platform
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunClock:
    start_monotonic: float

    @classmethod
    def start(cls) -> "RunClock":
        return cls(start_monotonic=time.perf_counter())

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_monotonic


class VerboseLogger:
    def __init__(self, run_name: str, clock: RunClock):
        self.run_name = run_name
        self.clock = clock

    def log(self, event: str, **fields) -> None:
        payload = {
            "ts_utc": utc_now_iso(),
            "run": self.run_name,
            "event": event,
            "elapsed_s": round(self.clock.elapsed_seconds(), 6),
            **fields,
        }
        print(json.dumps(payload, sort_keys=True), flush=True)


def get_memory_snapshot(label: str) -> Dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_mb = usage.ru_maxrss / 1024.0

    rss_mb = None
    statm = Path("/proc/self/statm")
    if statm.exists():
        values = statm.read_text().strip().split()
        if values:
            pages = int(values[1])
            rss_mb = pages * (os.sysconf("SC_PAGE_SIZE") / (1024 * 1024))

    snapshot = {
        "label": label,
        "rss_mb": round(rss_mb or 0.0, 3),
        "max_rss_mb": round(max_rss_mb, 3),
    }
    return snapshot


def percentile(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    index = (len(values) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(values[lower])
    weight = index - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def sequence_stats(lengths: Iterable[int]) -> Dict[str, float]:
    items = list(lengths)
    if not items:
        return {"min": 0, "mean": 0.0, "max": 0, "p95": 0.0}
    return {
        "min": int(min(items)),
        "mean": round(float(statistics.mean(items)), 3),
        "max": int(max(items)),
        "p95": round(percentile(sorted(items), 0.95), 3),
    }


def get_environment_snapshot(torch_module) -> Dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "torch": torch_module.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "omp_num_threads": os.getenv("OMP_NUM_THREADS", "unset"),
        "mkl_num_threads": os.getenv("MKL_NUM_THREADS", "unset"),
        "torch_num_threads": str(torch_module.get_num_threads()),
        "torch_num_interop_threads": str(torch_module.get_num_interop_threads()),
    }


def directory_metrics(path: str) -> Dict[str, int]:
    total_bytes = 0
    files = 0
    root = Path(path)
    for item in root.rglob("*"):
        if item.is_file():
            files += 1
            total_bytes += item.stat().st_size
    return {"file_count": files, "size_bytes": total_bytes}


def summarize_bottleneck(candidates: Dict[str, float]) -> str:
    if not candidates:
        return "unknown"
    top = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[0]
    return f"{top[0]} ({top[1]:.3f})"

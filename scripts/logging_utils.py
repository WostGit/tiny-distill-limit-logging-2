import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import psutil
import torch


def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


@dataclass
class RunTimer:
    start_time: float

    @classmethod
    def start(cls) -> "RunTimer":
        return cls(start_time=time.perf_counter())

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time


def log_section(title: str) -> None:
    print(f"\n{'=' * 28} {title} {'=' * 28}")


def get_memory_snapshot(tag: str) -> Dict[str, Any]:
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    data = {
        "tag": tag,
        "timestamp_utc": utc_ts(),
        "rss_mb": mem.rss / (1024 ** 2),
        "vms_mb": mem.vms / (1024 ** 2),
        "cpu_percent": psutil.cpu_percent(interval=None),
    }
    return data


def print_memory_snapshot(tag: str) -> Dict[str, Any]:
    snap = get_memory_snapshot(tag)
    print(
        "[MEM] "
        f"tag={snap['tag']} "
        f"rss_mb={snap['rss_mb']:.2f} "
        f"vms_mb={snap['vms_mb']:.2f} "
        f"cpu_percent={snap['cpu_percent']:.2f} "
        f"ts={snap['timestamp_utc']}"
    )
    return snap


def p95(values: Iterable[int]) -> float:
    arr = sorted(values)
    if not arr:
        return 0.0
    idx = int(0.95 * (len(arr) - 1))
    return float(arr[idx])


def summarize_lengths(name: str, lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        stats = {"min": 0.0, "mean": 0.0, "max": 0.0, "p95": 0.0}
    else:
        stats = {
            "min": float(min(lengths)),
            "mean": float(sum(lengths) / len(lengths)),
            "max": float(max(lengths)),
            "p95": p95(lengths),
        }
    print(
        f"[LEN] {name}: min={stats['min']:.1f} mean={stats['mean']:.1f} "
        f"max={stats['max']:.1f} p95={stats['p95']:.1f}"
    )
    return stats


def print_environment_info() -> Dict[str, Any]:
    env = {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    log_section("Environment")
    print(json.dumps(env, indent=2))
    return env

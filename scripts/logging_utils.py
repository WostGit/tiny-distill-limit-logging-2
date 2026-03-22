import json
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List

import psutil
import torch


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


@dataclass
class MemorySnapshot:
    tag: str
    ts_utc: str
    rss_bytes: int
    vms_bytes: int


class VerboseLogger:
    def __init__(self, run_name: str):
        self.run_name = run_name
        self.start_time = time.perf_counter()
        self.process = psutil.Process(os.getpid())
        self.memory_snapshots: List[MemorySnapshot] = []

    def log(self, message: str) -> None:
        elapsed = time.perf_counter() - self.start_time
        print(f"[{self.run_name}][+{elapsed:8.3f}s] {message}", flush=True)

    def snapshot_memory(self, tag: str) -> MemorySnapshot:
        mi = self.process.memory_info()
        snap = MemorySnapshot(tag=tag, ts_utc=utc_now_iso(), rss_bytes=mi.rss, vms_bytes=mi.vms)
        self.memory_snapshots.append(snap)
        self.log(
            f"MEMORY SNAPSHOT tag={tag} rss={_fmt_mb(mi.rss)} vms={_fmt_mb(mi.vms)}"
        )
        return snap

    def env_report(self) -> Dict[str, Any]:
        report = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS"),
            "pid": os.getpid(),
            "active_thread_count": threading.active_count(),
        }
        self.log("ENVIRONMENT REPORT START")
        for k, v in report.items():
            self.log(f"ENV {k}={v}")
        self.log("ENVIRONMENT REPORT END")
        return report

    def memory_as_dicts(self) -> List[Dict[str, Any]]:
        return [asdict(x) for x in self.memory_snapshots]


def compute_size_and_file_count(path: str) -> Dict[str, int]:
    total_bytes = 0
    file_count = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total_bytes += os.path.getsize(fp)
                file_count += 1
            except OSError:
                pass
    return {"bytes": total_bytes, "file_count": file_count}


def print_json_block(prefix: str, payload: Dict[str, Any]) -> None:
    print(f"{prefix} {json.dumps(payload, sort_keys=True)}", flush=True)

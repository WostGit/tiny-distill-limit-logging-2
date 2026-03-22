import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import psutil
import torch


@dataclass
class EventLogger:
    run_start: float = field(default_factory=time.perf_counter)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def now(self) -> float:
        return time.perf_counter()

    def since_start(self) -> float:
        return self.now() - self.run_start

    def log(self, name: str, **payload: Any) -> Dict[str, Any]:
        event = {
            "event": name,
            "t_rel_s": round(self.since_start(), 6),
            **payload,
        }
        self.events.append(event)
        print(f"[EVENT] {json.dumps(event, sort_keys=True)}", flush=True)
        return event


def bytes_to_mib(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 3)


def memory_snapshot(tag: str) -> Dict[str, Any]:
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    children_rss = 0
    for child in proc.children(recursive=True):
        try:
            children_rss += child.memory_info().rss
        except psutil.Error:
            pass
    payload = {
        "tag": tag,
        "rss_mib": bytes_to_mib(mem.rss),
        "vms_mib": bytes_to_mib(mem.vms),
        "children_rss_mib": bytes_to_mib(children_rss),
    }
    print(f"[MEMORY] {json.dumps(payload, sort_keys=True)}", flush=True)
    return payload


def environment_snapshot() -> Dict[str, Any]:
    payload = {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "env_omp_num_threads": os.getenv("OMP_NUM_THREADS"),
        "env_mkl_num_threads": os.getenv("MKL_NUM_THREADS"),
    }
    print(f"[ENV] {json.dumps(payload, sort_keys=True)}", flush=True)
    return payload


def checkpoint_dir_stats(path: str) -> Dict[str, Any]:
    p = Path(path)
    total_size = 0
    file_count = 0
    for child in p.rglob("*"):
        if child.is_file():
            total_size += child.stat().st_size
            file_count += 1
    payload = {
        "checkpoint_dir": str(p),
        "size_mib": bytes_to_mib(total_size),
        "file_count": file_count,
    }
    print(f"[CHECKPOINT] {json.dumps(payload, sort_keys=True)}", flush=True)
    return payload

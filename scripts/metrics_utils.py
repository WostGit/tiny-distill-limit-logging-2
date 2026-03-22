import json
from pathlib import Path
from typing import Any, Dict


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    ensure_parent(p)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    ensure_parent(p)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def summarize_times(seconds_by_name: Dict[str, float]) -> Dict[str, Any]:
    total = sum(seconds_by_name.values())
    summary = {}
    for name, seconds in seconds_by_name.items():
        pct = (seconds / total * 100.0) if total > 0 else 0.0
        summary[name] = {"seconds": round(seconds, 6), "pct_total": round(pct, 2)}
    summary["total_seconds"] = round(total, 6)
    return summary

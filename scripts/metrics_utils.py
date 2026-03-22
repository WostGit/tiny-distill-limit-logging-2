import json
from pathlib import Path
from typing import Any, Dict


class MetricsWriter:
    def __init__(self):
        self.data: Dict[str, Any] = {
            "events": [],
            "snapshots": [],
            "summary": {},
        }

    def add_event(self, event: Dict[str, Any]) -> None:
        self.data["events"].append(event)

    def add_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.data["snapshots"].append(snapshot)

    def set_summary(self, summary: Dict[str, Any]) -> None:
        self.data["summary"] = summary

    def write_json(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.data, indent=2, sort_keys=True))

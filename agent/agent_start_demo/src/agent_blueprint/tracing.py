from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from threading import Lock
from typing import Any


class JsonlTracer:
    def __init__(self, path: str | Path | None = None, *, echo: bool = False) -> None:
        self.path = Path(path) if path else None
        self.echo = echo
        self._lock = Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, *, run_id: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "run_id": run_id,
            "payload": payload,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        if self.echo:
            print(line)
        if self.path:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from uuid import uuid4


@dataclass(frozen=True)
class Asset:
    asset_type: str
    name: str
    payload: Dict[str, Any]
    asset_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class TraceEvent:
    component: str
    action: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    events: List[TraceEvent] = field(default_factory=list)

    def add(self, component: str, action: str, details: Dict[str, Any] | None = None) -> None:
        self.events.append(TraceEvent(component=component, action=action, details=details or {}))

    def component_names(self) -> List[str]:
        return [event.component for event in self.events]


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    output: str
    route: str
    trace: Trace
    assets: List[Asset] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionContext:
    user_id: str
    goal: str

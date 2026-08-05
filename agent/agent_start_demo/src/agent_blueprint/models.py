from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
import uuid


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class PlanStep:
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"


@dataclass(slots=True)
class Action:
    kind: Literal["tool", "finish", "replan"]
    reason: str
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None


@dataclass(slots=True)
class Observation:
    tool_name: str
    ok: bool
    output: Any = None
    error: str | None = None
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class Reflection:
    status: Literal["continue", "done", "replan", "blocked"]
    reason: str
    final_answer: str | None = None


@dataclass(slots=True)
class AgentState:
    goal: str
    namespace: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    status: Literal["new", "running", "completed", "blocked", "failed", "stopped"] = "new"
    plan: list[PlanStep] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    memories_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    step_count: int = 0
    tool_call_count: int = 0
    final_answer: str | None = None

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        copied = dict(data)
        copied["plan"] = [PlanStep(**item) for item in copied.get("plan", [])]
        copied["observations"] = [
            Observation(**item) for item in copied.get("observations", [])
        ]
        return cls(**copied)

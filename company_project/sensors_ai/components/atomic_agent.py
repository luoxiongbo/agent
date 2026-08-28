from __future__ import annotations

from typing import Protocol

from company_project.sensors_ai.core.context import ContextStore
from company_project.sensors_ai.core.result import Trace


class AtomicAgent(Protocol):
    name: str

    def execute(self, goal: str, context: ContextStore, trace: Trace) -> str:
        ...

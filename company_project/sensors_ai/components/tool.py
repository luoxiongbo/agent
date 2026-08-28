from __future__ import annotations

from typing import Protocol

from company_project.sensors_ai.core.context import ContextStore
from company_project.sensors_ai.core.result import Trace


class Tool(Protocol):
    name: str

    def run(self, goal: str, context: ContextStore, trace: Trace) -> str:
        ...

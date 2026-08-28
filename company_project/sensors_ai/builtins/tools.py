from __future__ import annotations

from company_project.sensors_ai.core.context import ContextStore
from company_project.sensors_ai.core.result import Trace


class BusinessDataTool:
    name = "business_data"

    def run(self, goal: str, context: ContextStore, trace: Trace) -> str:
        trace.add("tool:business_data", "invoke", {"goal": goal})
        context.audit_log.record("tool:business_data", "invoke", {"goal": goal})
        return "business data operation accepted"

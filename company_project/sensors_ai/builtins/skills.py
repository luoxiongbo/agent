from __future__ import annotations

from company_project.sensors_ai.components.skill import Skill
from company_project.sensors_ai.core.context import ContextStore
from company_project.sensors_ai.core.result import Asset, ExecutionResult, Trace


class SegmentBuildSkill(Skill):
    name = "segment_build"
    risk_action = "segment:create"

    def execute(self, goal: str, user_id: str, context: ContextStore, trace: Trace) -> ExecutionResult:
        trace.add("skill:segment_build", "execute", {"goal": goal})
        context.audit_log.record("skill:segment_build", "execute", {"goal": goal})
        business_tool = self.tools.get("business_data")
        if business_tool is not None:
            business_tool.run(goal, context, trace)
        asset = context.assets.add(
            Asset(
                asset_type="segment",
                name="AI generated segment",
                payload={"goal": goal, "created_by": user_id},
            )
        )
        output = f"已创建分群资产: {asset.name}"
        return ExecutionResult(
            status="completed",
            output=output,
            route="skill:segment_build",
            trace=trace,
            assets=[asset],
        )

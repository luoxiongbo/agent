import unittest

from company_project import (
    AssetStore,
    AuditLog,
    ContextStore,
    Governance,
    MainAgent,
    Registry,
)
from company_project.builtins import BusinessDataTool, SegmentBuildSkill


class GovernanceTest(unittest.TestCase):
    def test_governance_blocks_high_risk_action_without_permission(self):
        registry = Registry()
        registry.register_skill(SegmentBuildSkill(tools=[BusinessDataTool()]))
        governance = Governance(permissions={"guest": []})
        agent = MainAgent(
            registry=registry,
            context=ContextStore(assets=AssetStore(), audit_log=AuditLog()),
            governance=governance,
        )

        result = agent.run("帮我创建一个高价值用户分群", user_id="guest")

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.route, "governance")
        self.assertIn("permission", result.output)
        self.assertEqual(result.trace.component_names(), ["main_agent", "governance"])


if __name__ == "__main__":
    unittest.main()

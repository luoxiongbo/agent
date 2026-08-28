import unittest

from company_project import Asset, AssetStore, AuditLog, ContextStore


class ContextAndAssetsTest(unittest.TestCase):
    def test_context_store_keeps_knowledge_memory_assets_and_history_separate(self):
        assets = AssetStore()
        audit_log = AuditLog()
        context = ContextStore(
            knowledge={"指标口径": "首购转化率"},
            memories={"user:u1": "偏好简洁报告"},
            assets=assets,
            audit_log=audit_log,
        )

        context.record_history("u1", "分析新客转化", "完成分析")
        asset = assets.add(Asset(asset_type="report", name="新客转化分析", payload={"status": "ready"}))

        self.assertEqual(context.lookup_knowledge("指标"), {"指标口径": "首购转化率"})
        self.assertEqual(context.lookup_memory("user:u1"), "偏好简洁报告")
        self.assertEqual(context.history_for("u1")[0].goal, "分析新客转化")
        self.assertEqual(assets.list_by_type("report"), [asset])


    def test_audit_log_records_component_actions_in_order(self):
        audit_log = AuditLog()

        audit_log.record("main_agent", "receive_goal", {"goal": "x"})
        audit_log.record("skill:segment_build", "execute", {"asset": "segment"})

        self.assertEqual([entry.component for entry in audit_log.entries], ["main_agent", "skill:segment_build"])
        self.assertEqual(audit_log.entries[1].details, {"asset": "segment"})


if __name__ == "__main__":
    unittest.main()

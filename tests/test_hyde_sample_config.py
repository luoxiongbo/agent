import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "rag" / "hyDE" / "sample_demo.py"
ROOT = SOURCE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class HydeSampleConfigTest(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("rag.hyDE.sample_demo")

    def test_sample_demo_does_not_embed_api_key(self):
        source = SOURCE.read_text(encoding="utf-8")

        self.assertNotIn("sk-", source)

    def test_uses_zhipu_environment(self):
        with patch.dict(
            os.environ,
            {
                "ZHIPUAI_API_KEY": "zhipu-key",
                "ZHIPUAI_MODEL": "glm-5.2",
            },
            clear=True,
        ):
            self.assertEqual(self.module.get_api_key(), "zhipu-key")
            self.assertEqual(self.module.get_model(), "glm-5.2")

    def test_create_client_uses_zhipu_sdk(self):
        with patch.dict(
            os.environ,
            {"ZHIPUAI_API_KEY": "zhipu-key", "ZHIPUAI_MODEL": "glm-4.7"},
            clear=True,
        ), patch.object(self.module, "ZhipuAI") as zhipu_cls:
            client = self.module.create_client()

        self.assertIs(client, zhipu_cls.return_value)
        zhipu_cls.assert_called_once_with(
            api_key="zhipu-key",
        )

    def test_generate_hypothetical_doc_uses_chat_completions(self):
        calls = []
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="hypothetical answer")
                            )
                        ]
                    )
                )
            )
        )

        hypo_doc = self.module.generate_hypothetical_doc(
            "保险销售技巧",
            client,
            model="glm-4.7",
        )

        self.assertEqual(hypo_doc, "hypothetical answer")
        self.assertEqual(calls[0]["model"], "glm-4.7")
        self.assertIn("messages", calls[0])

    def test_builds_sample_index_from_demo_documents(self):
        class FakeEmbedModel:
            def encode(self, docs):
                return [[i, i + 1] for i, _ in enumerate(docs)]

        doc_embeddings, doc_ids = self.module.build_sample_index(FakeEmbedModel())

        self.assertEqual(len(doc_embeddings.tolist()), len(self.module.SAMPLE_DOCS))
        self.assertEqual(len(doc_ids), len(self.module.SAMPLE_DOCS))


if __name__ == "__main__":
    unittest.main()

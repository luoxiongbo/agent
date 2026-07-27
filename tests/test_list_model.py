import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class ListModelTest(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("rag.hyDE.list_model")

    def test_create_client_uses_api_key_and_base_url_from_environment(self):
        env = {
            "ZHIPUAI_API_KEY": "zhipu-key",
        }

        with patch.dict(os.environ, env, clear=True), patch.object(self.module, "ZhipuAI") as zhipu_cls:
            client = self.module.create_client()

        self.assertIs(client, zhipu_cls.return_value)
        zhipu_cls.assert_called_once_with(
            api_key="zhipu-key",
        )

    def test_list_model_ids_returns_zhipu_known_text_models(self):
        model_ids = self.module.list_model_ids()

        self.assertIn("glm-4.7", model_ids)
        self.assertIn("glm-4-plus", model_ids)

    def test_list_model_ids_can_be_overridden_from_environment(self):
        with patch.dict(os.environ, {"ZHIPUAI_MODELS": "glm-a, glm-b"}, clear=True):
            model_ids = self.module.list_model_ids()

        self.assertEqual(model_ids, ["glm-a", "glm-b"])


if __name__ == "__main__":
    unittest.main()

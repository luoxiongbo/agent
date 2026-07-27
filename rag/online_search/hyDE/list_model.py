import os

try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = None


KNOWN_TEXT_MODELS = [
    "glm-4.7",
    "glm-4.6",
    "glm-4.5",
    "glm-4.5-air",
    "glm-4.5-x",
    "glm-4.5-airx",
    "glm-4.5-flash",
    "glm-4-plus",
    "glm-4-air-250414",
    "glm-4-airx",
    "glm-4-flashx",
    "glm-4-flashx-250414",
]


def get_api_key() -> str:
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置 ZHIPUAI_API_KEY。")

    return api_key


def create_client():
    if ZhipuAI is None:
        raise RuntimeError("当前环境未安装 zhipuai，请先安装：pip install zhipuai")

    return ZhipuAI(api_key=get_api_key())


def list_model_ids():
    configured_models = os.getenv("ZHIPUAI_MODELS")
    if configured_models:
        return [
            model.strip()
            for model in configured_models.split(",")
            if model.strip()
        ]

    return KNOWN_TEXT_MODELS


def main():
    for model_id in list_model_ids():
        print(model_id)


if __name__ == "__main__":
    main()

"""
免费模型：
glm-4.7-flash
glm-4-flash-250414
glm-z1-flash
glm-4v-flash
glm-4.6v-flash
glm-4.1v-thinking-flash
"""
import os
from typing import Optional
from hello_agents import HelloAgentsLLM


class MyLLM(HelloAgentsLLM):
    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            provider: Optional[str] = "auto",
            **kwargs
    ):
        # 检查provider是否为我们想处理的'modelscope'
        if provider == "modelscope":
            print("正在使用自定义的 ModelScope Provider")

            # 解析 ModelScope 的凭证
            modelscope_api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            modelscope_base_url = base_url or "https://api-inference.modelscope.cn/v1/"

            # 验证凭证是否存在
            if not modelscope_api_key:
                raise ValueError("ModelScope API key not found. Please set MODELSCOPE_API_KEY environment variable.")

            # ModelScope 兼容 OpenAI 协议。仍由父类完成预算、适配器和资源状态初始化。
            super().__init__(
                model=model or os.getenv("MODELSCOPE_MODEL_ID") or "Qwen/Qwen3-8B",
                api_key=modelscope_api_key,
                base_url=modelscope_base_url,
                provider="openai",
                **kwargs,
            )
            self.provider = "modelscope"

        else:
            # 如果不是 modelscope, 则完全使用父类的原始逻辑来处理
            super().__init__(model=model, api_key=api_key, base_url=base_url, provider=provider, **kwargs)

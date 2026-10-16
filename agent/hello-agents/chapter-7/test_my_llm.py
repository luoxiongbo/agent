import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from my_llm import MyLLM


def test_modelscope_provider_initializes_the_base_llm_state():
    llm = MyLLM(
        provider="modelscope",
        api_key="test-key",
        temperature=0.2,
    )

    assert llm.provider == "modelscope"
    assert llm.budget is None
    assert llm._closed is False
    assert type(llm._adapter).__name__ == "OpenAIAdapter"
    assert llm.temperature == 0.2

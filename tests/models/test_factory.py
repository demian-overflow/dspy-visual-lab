from config.models import ModelConfig
from models.adapters.gemini import GeminiAdapter
from models.adapters.openrouter import OpenRouterAdapter
from models.factory import build_adapter


def test_build_adapter_gemini():
    config = ModelConfig(name="gemini-2.5-flash", provider="gemini")
    adapter = build_adapter(config, api_key="key", client=object())
    assert isinstance(adapter, GeminiAdapter)
    assert adapter.model == "gemini-2.5-flash"


def test_build_adapter_openrouter():
    config = ModelConfig(name="qwen/qwen2.5-vl", provider="openrouter")
    adapter = build_adapter(config, api_key="key", client=object())
    assert isinstance(adapter, OpenRouterAdapter)
    assert adapter.model == "qwen/qwen2.5-vl"


def test_build_adapter_unknown_provider_raises():
    config = ModelConfig(name="x", provider="nope")
    try:
        build_adapter(config, api_key="key", client=object())
        assert False, "expected ValueError"
    except ValueError:
        pass

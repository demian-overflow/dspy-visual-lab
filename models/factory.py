from config.models import ModelConfig
from models.adapters.base import BaseAdapter
from models.adapters.gemini import GeminiAdapter
from models.adapters.openrouter import OpenRouterAdapter


def build_adapter(config: ModelConfig, api_key: str, client) -> BaseAdapter:
    if config.provider == "gemini":
        return GeminiAdapter(client=client, api_key=api_key, model=config.name)
    if config.provider == "openrouter":
        return OpenRouterAdapter(client=client, api_key=api_key, model=config.name)
    raise ValueError(f"Unknown provider: {config.provider!r}")

from dataclasses import dataclass


@dataclass
class ModelConfig:


    name: str

    provider: str

    temperature: float = 0

    max_tokens: int = 4096



VISION_MODEL = ModelConfig(
    name="gemini-2.0-flash",
    provider="gemini"
)


PLANNER_MODEL = ModelConfig(
    name="qwen/qwen2.5-vl",
    provider="openrouter"
)

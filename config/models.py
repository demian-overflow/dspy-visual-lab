from dataclasses import dataclass


@dataclass
class ModelConfig:


    name: str

    provider: str

    temperature: float = 0

    max_tokens: int = 4096



# nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free is served from Nvidia's
# own free pool on OpenRouter (not Google AI Studio's shared free pool), so
# it doesn't hit the same "no billing = 0 quota" wall Gemini's direct API
# does, and doesn't require any billing account. Confirmed against the live
# OpenRouter API to return correct vision output at zero cost.
VISION_MODEL = ModelConfig(
    name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    provider="openrouter"
)


PLANNER_MODEL = ModelConfig(
    name="qwen/qwen2.5-vl",
    provider="openrouter"
)

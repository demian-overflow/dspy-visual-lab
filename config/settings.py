from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    model_config = ConfigDict(env_file=".env")

    app_name: str = "creative-lab"

    environment: str = "dev"


    openrouter_api_key: str | None = None

    gemini_api_key: str | None = None

    langfuse_public_key: str | None = None

    langfuse_secret_key: str | None = None

    langfuse_host: str = "https://cloud.langfuse.com"


    runs_dir: str = "runs"


    max_agent_iterations: int = 5


    def api_key_for(self, provider: str) -> str | None:
        """Look up the right API key for a ModelConfig.provider value.

        Call sites used to hardcode `settings.gemini_api_key` regardless of
        which provider a ModelConfig actually pointed at -- harmless while
        every model happened to be Gemini, but silently wrong (wrong key
        sent to the wrong provider) the moment one model config uses
        OpenRouter instead.
        """
        return {
            "gemini": self.gemini_api_key,
            "openrouter": self.openrouter_api_key,
        }.get(provider)



settings = Settings()

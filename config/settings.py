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



settings = Settings()

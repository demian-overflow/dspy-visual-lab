from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    app_name: str = "creative-lab"

    environment: str = "dev"


    openrouter_api_key: str | None = None

    gemini_api_key: str | None = None


    runs_dir: str = "runs"


    max_agent_iterations: int = 5


    class Config:

        env_file = ".env"



settings = Settings()

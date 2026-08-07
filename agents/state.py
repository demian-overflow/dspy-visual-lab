from pydantic import BaseModel, Field


class AgentState(BaseModel):

    task_id: str

    image: str

    scene: dict = Field(
        default_factory=dict
    )

    plan: list = Field(
        default_factory=list
    )

    tool_results: list = Field(
        default_factory=list
    )

    artifact: dict = Field(
        default_factory=dict
    )

    iteration: int = 0

    finished: bool = False

    score: dict = Field(
        default_factory=dict
    )

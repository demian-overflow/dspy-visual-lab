from pydantic import BaseModel


class Experiment(BaseModel):

    name: str

    description: str = ""


    parser_model: str

    planner_model: str


    optimizer: str | None = None


    max_iterations: int = 5


    metrics: dict = {}

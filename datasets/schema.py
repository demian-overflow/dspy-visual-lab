from pydantic import BaseModel


class CreativeSample(BaseModel):

    id: str

    image_path: str

    category: str

    difficulty: str = "unknown"


    scene_path: str | None = None


    reference_output: str | None = None


    tags: list[str] = []


    metadata: dict = {}

from pydantic import BaseModel, Field
from typing import List, Dict


class BoundingBox(BaseModel):

    x: float
    y: float
    width: float
    height: float



class Color(BaseModel):

    hex: str
    name: str | None = None



class TextElement(BaseModel):

    id: str

    content: str

    bbox: BoundingBox

    font_family: str | None = None

    font_size: float | None = None

    font_weight: str | None = None

    color: str | None = None



class VisualObject(BaseModel):

    id: str

    type: str

    bbox: BoundingBox

    description: str | None = None

    attributes: Dict = Field(
        default_factory=dict
    )



class Layout(BaseModel):

    alignment: str | None = None

    grid: str | None = None

    spacing: Dict = Field(
        default_factory=dict
    )



class Scene(BaseModel):

    width: int

    height: int

    background: str = "#ffffff"

    objects: List[VisualObject] = []

    text: List[TextElement] = []

    colors: List[Color] = []

    layout: Layout = Layout()

    metadata: Dict = Field(
        default_factory=dict
    )

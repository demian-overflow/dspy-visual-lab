from pydantic import BaseModel
from datetime import datetime


class Event(BaseModel):

    timestamp: datetime

    name: str

    run_id: str

    data: dict = {}

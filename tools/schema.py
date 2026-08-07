from pydantic import BaseModel


class ToolCall(BaseModel):
    name: str
    arguments: dict


class ToolResult(BaseModel):
    name: str
    output: dict
    success: bool = True

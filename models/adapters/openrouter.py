import base64
import mimetypes

from .base import BaseAdapter


def _as_image_url(image: str) -> str:
    """Accept either a ready-to-use URL/data-URI, or a local file path.

    OpenRouter's chat-completions API takes `image_url.url` as either an
    http(s) URL or a `data:` URI -- it never reads a bare filesystem path.
    Tools like `tools/vision/ocr.py` call `adapter.vision(local_path, ...)`
    with a plain path, so this mirrors GeminiAdapter.vision()'s base64
    encoding rather than sending the model a string it can't fetch.
    """
    if image.startswith(("http://", "https://", "data:")):
        return image

    with open(image, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    mime_type = mimetypes.guess_type(str(image))[0] or "image/png"
    return f"data:{mime_type};base64,{encoded}"


class OpenRouterAdapter(BaseAdapter):

    URL = (
        "https://openrouter.ai/api/v1/"
        "chat/completions"
    )


    def __init__(
        self,
        client,
        api_key,
        model
    ):
        self.client = client
        self.api_key = api_key
        self.model = model


    async def generate(
        self,
        messages,
        **kwargs
    ):

        return await self.client.post(
            self.URL,
            headers={
                "Authorization":
                    f"Bearer {self.api_key}",
                "Content-Type":
                    "application/json"
            },
            payload={
                "model": self.model,
                "messages": messages,
                **kwargs
            }
        )


    async def vision(
        self,
        image,
        prompt,
        **kwargs
    ):

        return await self.generate(
            [
                {
                    "role":"user",
                    "content":[
                        {
                            "type":"text",
                            "text":prompt
                        },
                        {
                            "type":"image_url",
                            "image_url":{
                                "url":_as_image_url(image)
                            }
                        }
                    ]
                }
            ],
            **kwargs
        )


    async def tool_call(
        self,
        messages,
        tools,
        **kwargs
    ):

        return await self.generate(
            messages,
            tools=tools,
            tool_choice="auto",
            **kwargs
        )

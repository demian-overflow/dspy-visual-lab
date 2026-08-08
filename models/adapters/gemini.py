import base64

from .base import BaseAdapter


class GeminiAdapter(BaseAdapter):

    def __init__(self, client, api_key, model="gemini-2.5-flash"):
        self.client = client
        self.api_key = api_key
        self.model = model

    def url(self):
        return (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model}:generateContent?key={self.api_key}"
        )

    async def generate(self, contents, **kwargs):
        return await self.client.post(
            self.url(),
            payload={"contents": contents, **kwargs},
        )

    async def vision(self, image, prompt, **kwargs):
        with open(image, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")

        return await self.generate(
            [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": encoded,
                            }
                        },
                    ]
                }
            ],
            **kwargs,
        )

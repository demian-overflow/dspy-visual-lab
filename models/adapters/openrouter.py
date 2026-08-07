from .base import BaseAdapter


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
                                "url":image
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

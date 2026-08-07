import aiohttp
import orjson


class HTTPClient:

    def __init__(
        self,
        timeout=120
    ):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=timeout
            ),
            json_serialize=orjson.dumps
        )


    async def post(
        self,
        url,
        headers=None,
        payload=None
    ):

        async with self.session.post(
            url,
            headers=headers,
            json=payload
        ) as response:

            body = await response.read()

            if response.status >= 400:
                raise Exception(
                    body.decode()
                )

            return orjson.loads(body)


    async def close(self):
        await self.session.close()

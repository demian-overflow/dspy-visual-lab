import asyncio

import aiohttp
import orjson


def _json_serialize(obj):
    # aiohttp wants `str` from json_serialize; orjson returns `bytes`.
    return orjson.dumps(obj).decode()


class HTTPClient:

    def __init__(
        self,
        timeout=120
    ):
        self.timeout = timeout
        # An aiohttp.ClientSession binds to the event loop that is running when
        # it is created, so sessions are made lazily (and per loop) instead of
        # in __init__ -- constructing HTTPClient() outside a loop used to raise
        # "RuntimeError: no running event loop".
        self._sessions = {}


    def session(self):
        loop = asyncio.get_running_loop()
        session = self._sessions.get(loop)

        if session is None or session.closed:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.timeout
                ),
                json_serialize=_json_serialize
            )
            self._sessions[loop] = session

        return session


    async def post(
        self,
        url,
        headers=None,
        payload=None
    ):

        async with self.session().post(
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
        current = asyncio.get_running_loop()

        for loop, session in list(self._sessions.items()):

            if session.closed:
                continue

            if loop is current:
                await session.close()
            elif loop.is_running():
                await asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(session.close(), loop)
                )

        self._sessions.clear()

import asyncio
import threading

import dspy

_loop = None
_loop_lock = threading.Lock()


def _background_loop():
    """A long-lived event loop running on a daemon thread.

    Sync `forward()` cannot use `asyncio.run()`: DSPy modules are routinely
    called synchronously from inside async code (e.g. `CreativeAgent.run`),
    where that raises "asyncio.run() cannot be called from a running event
    loop". Driving every sync call from one background loop works in both
    cases, and keeps all adapter I/O (including the aiohttp session) on a
    single, stable loop.
    """
    global _loop

    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(
                target=_loop.run_forever,
                name="adapter-lm-loop",
                daemon=True,
            ).start()

    return _loop


class _Message:
    def __init__(self, content, role="assistant"):
        self.role = role
        self.content = content
        self.tool_calls = None


class _Choice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason
        self.logprobs = None


class _Response:
    """Minimal OpenAI-shaped response object.

    DSPy's `BaseLM._process_completion` reads `response.choices[0].message.content`
    by *attribute*, so a plain dict is not a valid legacy `forward()` return value.
    """

    def __init__(self, text, model):
        self.choices = [_Choice(_Message(text))]
        self.model = model
        self.usage = None
        self.id = None
        self.cache_hit = False


class AdapterLM(dspy.LM):

    forward_contract = "legacy"

    def __init__(self, adapter, model_name):
        super().__init__(model=model_name, cache=False)
        self.adapter = adapter
        self.model = model_name

    @staticmethod
    def _extract_text(raw_response):
        candidates = raw_response.get("candidates")
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)

        choices = raw_response.get("choices")
        if choices:
            return choices[0].get("message", {}).get("content", "")

        raise ValueError(f"Unrecognized adapter response shape: {raw_response!r}")

    async def aforward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]

        raw_response = await self.adapter.generate(messages, **kwargs)

        return _Response(self._extract_text(raw_response), self.model)

    def forward(self, prompt=None, messages=None, **kwargs):
        coroutine = self.aforward(prompt=prompt, messages=messages, **kwargs)

        return asyncio.run_coroutine_threadsafe(coroutine, _background_loop()).result()

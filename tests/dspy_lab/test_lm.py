import pytest

from dspy_lab.lm import AdapterLM


class FakeAdapter:
    def __init__(self, reply):
        self.reply = reply
        self.last_call = None

    async def generate(self, messages, **kwargs):
        self.last_call = messages
        return {"candidates": [{"content": {"parts": [{"text": self.reply}]}}]}


@pytest.mark.asyncio
async def test_aforward_returns_openai_shaped_response_with_adapter_text():
    adapter = FakeAdapter(reply="hello from adapter")
    lm = AdapterLM(adapter=adapter, model_name="fake-model")

    result = await lm.aforward(messages=[{"role": "user", "content": "hi"}])

    assert result["choices"][0]["message"]["content"] == "hello from adapter"
    assert adapter.last_call == [{"role": "user", "content": "hi"}]


def test_forward_is_sync_wrapper_over_aforward():
    adapter = FakeAdapter(reply="sync path")
    lm = AdapterLM(adapter=adapter, model_name="fake-model")

    result = lm.forward(messages=[{"role": "user", "content": "hi"}])

    assert result["choices"][0]["message"]["content"] == "sync path"

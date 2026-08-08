import asyncio

import pytest

from dspy_lab.lm import AdapterLM


class FakeAdapter:
    def __init__(self, reply):
        self.reply = reply
        self.last_call = None

    async def generate(self, messages, **kwargs):
        self.last_call = messages
        return {"candidates": [{"content": {"parts": [{"text": self.reply}]}}]}


def test_call_goes_through_dspy_and_returns_adapter_text():
    """Exercise dspy.LM.__call__, which requires an attribute-access response."""
    adapter = FakeAdapter(reply="hello from adapter")
    lm = AdapterLM(adapter=adapter, model_name="fake-model")

    outputs = lm(messages=[{"role": "user", "content": "hi"}])

    assert outputs == ["hello from adapter"]
    assert adapter.last_call == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_acall_goes_through_dspy_and_returns_adapter_text():
    adapter = FakeAdapter(reply="async hello")
    lm = AdapterLM(adapter=adapter, model_name="fake-model")

    outputs = await lm.acall(messages=[{"role": "user", "content": "hi"}])

    assert outputs == ["async hello"]


@pytest.mark.asyncio
async def test_aforward_returns_object_with_attribute_access():
    adapter = FakeAdapter(reply="hello from adapter")
    lm = AdapterLM(adapter=adapter, model_name="fake-model")

    response = await lm.aforward(messages=[{"role": "user", "content": "hi"}])

    # DSPy reads these by attribute, not by key.
    assert response.choices[0].message.content == "hello from adapter"
    assert response.choices[0].finish_reason == "stop"
    assert response.model == "fake-model"


def test_forward_works_outside_an_event_loop():
    adapter = FakeAdapter(reply="sync path")
    lm = AdapterLM(adapter=adapter, model_name="fake-model")

    response = lm.forward(messages=[{"role": "user", "content": "hi"}])

    assert response.choices[0].message.content == "sync path"


def test_sync_call_works_from_inside_a_running_event_loop():
    """Regression: `asyncio.run()` in forward() blew up under async callers."""
    adapter = FakeAdapter(reply="from async context")
    lm = AdapterLM(adapter=adapter, model_name="fake-model")

    async def caller():
        return lm(messages=[{"role": "user", "content": "hi"}])

    assert asyncio.run(caller()) == ["from async context"]

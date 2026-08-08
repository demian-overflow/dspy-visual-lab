import base64

import pytest

from models.adapters.gemini import GeminiAdapter, build_request


class CapturingClient:
    def __init__(self):
        self.calls = []

    async def post(self, url, headers=None, payload=None):
        self.calls.append({"url": url, "headers": headers, "payload": payload})
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}


def test_build_request_moves_system_message_to_system_instruction():
    payload = build_request(
        [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
        ]
    )

    assert payload["system_instruction"] == {"parts": [{"text": "be concise"}]}
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert all(c["role"] != "system" for c in payload["contents"])


def test_build_request_maps_assistant_role_to_model():
    payload = build_request([{"role": "assistant", "content": "prior turn"}])

    assert payload["contents"][0]["role"] == "model"


def test_build_request_converts_image_data_uri_to_inline_data():
    encoded = base64.b64encode(b"fake-bytes").decode()
    payload = build_request(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ],
            }
        ]
    )

    parts = payload["contents"][0]["parts"]
    assert parts[0] == {"text": "what is this"}
    assert parts[1]["inline_data"] == {"mime_type": "image/jpeg", "data": encoded}


def test_build_request_puts_generation_params_in_generation_config():
    payload = build_request(
        [{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=512,
        response_format={"type": "json_object"},
        cache=False,
        num_retries=3,
    )

    assert payload["generationConfig"] == {
        "temperature": 0.2,
        "maxOutputTokens": 512,
        "response_mime_type": "application/json",
    }
    # DSPy plumbing must not leak into the Gemini request body.
    assert "response_format" not in payload
    assert "cache" not in payload
    assert "num_retries" not in payload["generationConfig"]


@pytest.mark.asyncio
async def test_generate_sends_api_key_as_header_not_query_string():
    client = CapturingClient()
    adapter = GeminiAdapter(client=client, api_key="secret-key", model="gemini-2.5-flash")

    await adapter.generate([{"role": "user", "content": "hi"}])

    call = client.calls[0]
    assert "secret-key" not in call["url"]
    assert "key=" not in call["url"]
    assert call["headers"]["x-goog-api-key"] == "secret-key"


@pytest.mark.asyncio
async def test_vision_inlines_image_bytes_with_detected_mime_type(two_color_image):
    client = CapturingClient()
    adapter = GeminiAdapter(client=client, api_key="k", model="gemini-2.5-flash")

    await adapter.vision(str(two_color_image), prompt="describe")

    parts = client.calls[0]["payload"]["contents"][0]["parts"]
    assert parts[0] == {"text": "describe"}
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert base64.b64decode(parts[1]["inline_data"]["data"]) == two_color_image.read_bytes()

import base64
import mimetypes
import re

from .base import BaseAdapter

_DATA_URI = re.compile(r"^data:(?P<mime>[^;,]+);base64,(?P<data>.*)$", re.DOTALL)

# OpenAI-style generation parameter -> Gemini generationConfig field.
_GENERATION_CONFIG_KEYS = {
    "temperature": "temperature",
    "top_p": "topP",
    "top_k": "topK",
    "max_tokens": "maxOutputTokens",
    "max_output_tokens": "maxOutputTokens",
    "max_completion_tokens": "maxOutputTokens",
    "n": "candidateCount",
    "stop": "stopSequences",
    "presence_penalty": "presencePenalty",
    "frequency_penalty": "frequencyPenalty",
}

_ROLES = {"user": "user", "assistant": "model", "model": "model", "tool": "user"}


def _wants_json(response_format):
    """Whether a DSPy/OpenAI `response_format` value asks for JSON output."""
    if response_format is None:
        return False
    if isinstance(response_format, dict):
        return response_format.get("type") in ("json_object", "json_schema")
    # A pydantic model class (DSPy's structured-output hint) also means JSON.
    return True


def _part_from_content_item(item):
    """Convert one OpenAI-style content part into a Gemini part."""
    if isinstance(item, str):
        return {"text": item}

    kind = item.get("type")

    if kind == "text":
        return {"text": item.get("text", "")}

    if kind == "image_url":
        url = item["image_url"]["url"] if isinstance(item.get("image_url"), dict) else item.get("image_url")
        match = _DATA_URI.match(url or "")
        if match:
            return {
                "inline_data": {
                    "mime_type": match.group("mime"),
                    "data": match.group("data"),
                }
            }
        return {"file_data": {"file_uri": url}}

    if "text" in item:
        return {"text": item["text"]}

    raise ValueError(f"Unsupported message content part: {item!r}")


def _parts_from_content(content):
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}]
    return [_part_from_content_item(item) for item in content]


def build_request(messages, **kwargs):
    """Translate OpenAI-style chat messages + kwargs into a Gemini request body.

    Gemini's `generateContent` wants `contents: [{role, parts: [...]}]`, keeps the
    system prompt in a separate top-level `system_instruction`, and takes
    generation parameters under `generationConfig` -- none of which match the
    OpenAI chat shape DSPy hands to an LM.
    """
    contents = []
    system_parts = []

    for message in messages or []:
        role = message.get("role", "user")
        parts = _parts_from_content(message.get("content"))
        if not parts:
            continue
        if role == "system":
            system_parts.extend(parts)
        else:
            contents.append({"role": _ROLES.get(role, "user"), "parts": parts})

    payload = {"contents": contents}

    if system_parts:
        payload["system_instruction"] = {"parts": system_parts}

    generation_config = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in _GENERATION_CONFIG_KEYS:
            generation_config[_GENERATION_CONFIG_KEYS[key]] = value
        elif key == "response_format" and _wants_json(value):
            generation_config["response_mime_type"] = "application/json"
        elif key == "generationConfig" and isinstance(value, dict):
            generation_config.update(value)
        # Anything else (cache, num_retries, api_* ...) is DSPy plumbing, not a
        # Gemini request field, and is dropped: Gemini rejects unknown keys.

    if generation_config:
        payload["generationConfig"] = generation_config

    return payload


class GeminiAdapter(BaseAdapter):

    def __init__(self, client, api_key, model="gemini-2.5-flash"):
        self.client = client
        self.api_key = api_key
        self.model = model

    def url(self):
        return (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model}:generateContent"
        )

    def headers(self):
        # The key goes in a header, never the query string, so it cannot leak
        # into access logs or proxy traces.
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    async def post(self, payload):
        return await self.client.post(self.url(), headers=self.headers(), payload=payload)

    async def generate(self, messages, **kwargs):
        return await self.post(build_request(messages, **kwargs))

    async def vision(self, image, prompt, **kwargs):
        with open(image, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")

        mime_type = mimetypes.guess_type(str(image))[0] or "image/png"

        return await self.generate(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                        },
                    ],
                }
            ],
            **kwargs,
        )

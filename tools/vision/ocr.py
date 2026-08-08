import json

from config.models import VISION_MODEL
from config.settings import settings
from models.adapters.aiohttp_client import HTTPClient
from models.factory import build_adapter

from ..registry import register

_adapter = None
_client = None


def _get_adapter():
    global _adapter, _client
    if _adapter is None:
        _client = HTTPClient()
        _adapter = build_adapter(VISION_MODEL, api_key=settings.gemini_api_key, client=_client)
    return _adapter


def _extract_text(raw_response):
    try:
        return raw_response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"ocr: unexpected Gemini response shape (no candidates[0].content.parts[0].text): "
            f"{raw_response!r}"
        ) from exc


@register(
    "ocr",
    "Extract text with coordinates"
)
async def ocr(image):
    adapter = _get_adapter()
    raw_response = await adapter.vision(
        image,
        prompt=(
            "Return ONLY JSON matching "
            '{"text": [strings], "boxes": [{"x":num,"y":num,"width":num,"height":num}]} '
            "for every text element visible in this image, in pixel coordinates."
        ),
    )
    text = _extract_text(raw_response)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ocr: model did not return valid JSON: {text!r}") from exc

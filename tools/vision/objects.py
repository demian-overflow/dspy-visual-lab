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
            f"detect_objects: unexpected Gemini response shape "
            f"(no candidates[0].content.parts[0].text): {raw_response!r}"
        ) from exc


@register(
    "detect_objects",
    "Detect objects and bounding boxes"
)
async def detect_objects(image):
    adapter = _get_adapter()
    raw_response = await adapter.vision(
        image,
        prompt=(
            "Return ONLY JSON matching "
            '{"objects": [{"type": string, "bbox": {"x":num,"y":num,"width":num,"height":num}}]} '
            "for every salient visual object (not text) in this image, in pixel coordinates."
        ),
    )
    text = _extract_text(raw_response)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"detect_objects: model did not return valid JSON: {text!r}") from exc

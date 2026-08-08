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
        _adapter = build_adapter(
            VISION_MODEL, api_key=settings.api_key_for(VISION_MODEL.provider), client=_client
        )
    return _adapter


def _extract_text(raw_response):
    # Gemini shapes the response as candidates[0].content.parts[*].text;
    # OpenRouter (and other OpenAI-compatible providers) use
    # choices[0].message.content. VISION_MODEL.provider decides which
    # adapter -- and therefore which shape -- is actually in play.
    try:
        candidates = raw_response.get("candidates")
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)

        choices = raw_response.get("choices")
        if choices:
            return choices[0].get("message", {}).get("content", "")
    except (AttributeError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"detect_objects: unrecognized model response shape: {raw_response!r}"
        ) from exc

    raise ValueError(f"detect_objects: unrecognized model response shape: {raw_response!r}")


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

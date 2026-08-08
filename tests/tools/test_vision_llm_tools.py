import json
from unittest.mock import AsyncMock, patch

import pytest

from tools.vision.ocr import ocr
from tools.vision.objects import detect_objects


def _gemini_response(payload: dict):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}


@pytest.mark.asyncio
async def test_ocr_parses_gemini_json_response(two_color_image):
    fake_vision = AsyncMock(
        return_value=_gemini_response(
            {"text": ["Hello"], "boxes": [{"x": 1, "y": 2, "width": 3, "height": 4}]}
        )
    )
    with patch("tools.vision.ocr._get_adapter") as get_adapter:
        get_adapter.return_value.vision = fake_vision
        result = await ocr(str(two_color_image))

    assert result["text"] == ["Hello"]
    assert result["boxes"] == [{"x": 1, "y": 2, "width": 3, "height": 4}]


@pytest.mark.asyncio
async def test_detect_objects_parses_gemini_json_response(two_color_image):
    fake_vision = AsyncMock(
        return_value=_gemini_response(
            {"objects": [{"type": "logo", "bbox": {"x": 0, "y": 0, "width": 5, "height": 5}}]}
        )
    )
    with patch("tools.vision.objects._get_adapter") as get_adapter:
        get_adapter.return_value.vision = fake_vision
        result = await detect_objects(str(two_color_image))

    assert result["objects"][0]["type"] == "logo"


@pytest.mark.asyncio
async def test_ocr_raises_clear_error_on_non_json_response(two_color_image):
    fake_vision = AsyncMock(
        return_value={"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}
    )
    with patch("tools.vision.ocr._get_adapter") as get_adapter:
        get_adapter.return_value.vision = fake_vision
        with pytest.raises(ValueError, match="did not return valid JSON"):
            await ocr(str(two_color_image))


@pytest.mark.asyncio
async def test_detect_objects_raises_clear_error_on_malformed_response_shape(two_color_image):
    fake_vision = AsyncMock(return_value={"unexpected": "shape"})
    with patch("tools.vision.objects._get_adapter") as get_adapter:
        get_adapter.return_value.vision = fake_vision
        with pytest.raises(ValueError, match="unexpected Gemini response shape"):
            await detect_objects(str(two_color_image))

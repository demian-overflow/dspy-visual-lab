"""Integration test against the real, live VISION_MODEL provider.

Skipped unless a real API key is configured for whichever provider
VISION_MODEL currently points at. This is the only test that exercises the
full LM path (DSPy -> AdapterLM -> adapter -> HTTP), which is where
message-shape and response-shape bugs hide.
"""

import asyncio
import json
from pathlib import Path

import dspy
import pytest

from config.models import VISION_MODEL
from config.settings import settings
from dspy_lab.lm import AdapterLM
from dspy_lab.modules.scene_parser import SceneParser
from models.adapters.aiohttp_client import HTTPClient
from models.factory import build_adapter
from scene.parser import SceneParser as SceneModelParser

IMAGE = Path(__file__).resolve().parents[2] / "datasets" / "raw" / "images" / "poster_002.jpg"

_PLACEHOLDER_KEYS = (None, "", "xxx")

requires_vision_model_key = pytest.mark.skipif(
    settings.api_key_for(VISION_MODEL.provider) in _PLACEHOLDER_KEYS,
    reason=f"no API key configured for VISION_MODEL.provider={VISION_MODEL.provider!r}; "
    "skipping live integration test",
)


@requires_vision_model_key
def test_scene_parser_returns_valid_scene_from_real_image():
    client = HTTPClient()
    adapter = build_adapter(VISION_MODEL, api_key=settings.api_key_for(VISION_MODEL.provider), client=client)

    with dspy.context(lm=AdapterLM(adapter=adapter, model_name=VISION_MODEL.name)):
        try:
            prediction = SceneParser()(image=dspy.Image.from_path(str(IMAGE)))
        finally:
            asyncio.run(client.close())

    raw_scene = prediction.scene
    scene_dict = json.loads(raw_scene) if isinstance(raw_scene, str) else raw_scene

    scene = SceneModelParser().from_json(scene_dict)

    assert scene.width > 0
    assert scene.height > 0
    assert scene.text or scene.objects

import pytest

from tools.vision.image_analysis import analyze_image
from tools.design.palette import extract_palette


@pytest.mark.asyncio
async def test_analyze_image_returns_dimensions(two_color_image):
    result = await analyze_image(str(two_color_image))
    assert result["width"] == 200
    assert result["height"] == 100
    assert result["aspect_ratio"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_extract_palette_finds_dominant_colors(two_color_image):
    result = await extract_palette(str(two_color_image), n=2)
    colors = {c.lower() for c in result["colors"]}
    assert "#ff0000" in colors
    assert "#0000ff" in colors

import pytest

from tools.generation.svg import generate_svg
from tools.generation.screenshot import rasterize_svg


@pytest.mark.asyncio
async def test_generate_svg_includes_text_and_background():
    scene = {
        "width": 100,
        "height": 50,
        "background": "#ffffff",
        "objects": [],
        "text": [
            {
                "id": "t1",
                "content": "Hello",
                "bbox": {"x": 5, "y": 5, "width": 40, "height": 10},
                "font_family": "sans-serif",
                "font_size": 12,
                "color": "#000000",
            }
        ],
    }
    result = await generate_svg(scene)
    assert "<svg" in result["svg"]
    assert "Hello" in result["svg"]
    assert 'width="100"' in result["svg"]


@pytest.mark.asyncio
async def test_rasterize_svg_writes_a_real_png(tmp_path, monkeypatch):
    import config.paths as cfg_paths
    monkeypatch.setattr(cfg_paths, "ARTIFACTS", tmp_path / "artifacts")

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        '<rect width="10" height="10" fill="red"/></svg>'
    )
    result = await rasterize_svg(svg)
    image_path = result["image_path"]

    from PIL import Image
    with Image.open(image_path) as img:
        assert img.size == (10, 10)

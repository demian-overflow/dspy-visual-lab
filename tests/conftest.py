from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def two_color_image(tmp_path: Path) -> Path:
    img = Image.new("RGB", (200, 100), color="#ff0000")
    for x in range(100, 200):
        for y in range(100):
            img.putpixel((x, y), (0, 0, 255))
    path = tmp_path / "two_color.png"
    img.save(path)
    return path

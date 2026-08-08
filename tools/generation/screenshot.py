import uuid

import cairosvg

from config.paths import ARTIFACTS
from ..registry import register


@register(
    "rasterize_svg",
    "Rasterize an SVG string to a PNG file"
)
async def rasterize_svg(svg):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    image_path = ARTIFACTS / f"{uuid.uuid4()}.png"
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(image_path))

    return {"image_path": str(image_path)}

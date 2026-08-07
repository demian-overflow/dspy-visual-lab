from PIL import Image

from ..registry import register


@register(
    "analyze_image",
    "General image properties"
)
async def analyze_image(image):
    with Image.open(image) as img:
        width, height = img.size

    return {
        "width": width,
        "height": height,
        "aspect_ratio": width / height if height else 0,
    }

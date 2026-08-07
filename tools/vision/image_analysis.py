from ..registry import register


@register(
    "analyze_image",
    "General image properties"
)
async def analyze_image(image):

    return {
        "width":0,
        "height":0,
        "aspect_ratio":0
    }

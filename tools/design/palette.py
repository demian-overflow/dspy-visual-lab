from ..registry import register


@register(
    "extract_palette",
    "Extract dominant colors"
)
async def extract_palette(image):

    return {
        "colors":[]
    }

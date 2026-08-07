from ..registry import register


@register(
    "ocr",
    "Extract text with coordinates"
)
async def ocr(image):

    return {
        "text": [],
        "boxes": []
    }

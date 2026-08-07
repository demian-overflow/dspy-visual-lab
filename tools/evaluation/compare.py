from ..registry import register


@register(
    "compare_images",
    "Compare two images"
)
async def compare_images(
    original,
    generated
):

    return {
        "similarity":0.0
    }

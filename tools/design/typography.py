from ..registry import register


@register(
    "estimate_typography",
    "Estimate fonts and typography"
)
async def estimate_typography(image):

    return {
        "font_family":"",
        "weight":"",
        "size":0
    }

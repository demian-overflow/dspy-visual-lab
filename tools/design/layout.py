from ..registry import register


@register(
    "analyze_layout",
    "Analyze composition and spacing"
)
async def analyze_layout(image):

    return {
        "grid":"",
        "alignment":"",
        "spacing":{}
    }

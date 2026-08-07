from ..registry import register


@register(
    "detect_objects",
    "Detect objects and bounding boxes"
)
async def detect_objects(image):

    return {
        "objects": []
    }

from ..registry import register


@register(
    "score_scene",
    "Score scene reconstruction"
)
async def score_scene(
    original,
    generated
):

    return {
        "layout":0,
        "text":0,
        "colors":0,
        "overall":0
    }

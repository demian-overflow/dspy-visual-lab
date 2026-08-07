from ..registry import register


@register(
    "generate_svg",
    "Generate SVG design"
)
async def generate_svg(scene):

    return {
        "svg":""
    }

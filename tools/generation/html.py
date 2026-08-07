from ..registry import register


@register(
    "generate_html",
    "Generate HTML/CSS recreation"
)
async def generate_html(scene):

    return {
        "html":"",
        "css":""
    }

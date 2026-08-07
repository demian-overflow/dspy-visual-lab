from ..registry import register


@register(
    "render_browser",
    "Render HTML screenshot"
)
async def render_browser(html):

    return {
        "image_path":""
    }

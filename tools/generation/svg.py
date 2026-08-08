from xml.sax.saxutils import escape

from ..registry import register


def _rect_for(obj, fill="#cccccc"):
    b = obj["bbox"]
    return f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["width"]}" height="{b["height"]}" fill="{fill}"/>'


def _text_for(item):
    b = item["bbox"]
    family = item.get("font_family") or "sans-serif"
    size = item.get("font_size") or 16
    color = item.get("color") or "#000000"
    content = escape(item.get("content", ""))
    return (
        f'<text x="{b["x"]}" y="{b["y"] + size}" '
        f'font-family="{escape(family)}" font-size="{size}" fill="{color}">{content}</text>'
    )


@register(
    "generate_svg",
    "Generate SVG design"
)
async def generate_svg(scene):
    width = scene.get("width", 0)
    height = scene.get("height", 0)
    background = scene.get("background", "#ffffff")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="{background}"/>',
    ]
    parts += [_rect_for(obj) for obj in scene.get("objects", [])]
    parts += [_text_for(item) for item in scene.get("text", [])]
    parts.append("</svg>")

    return {"svg": "".join(parts)}

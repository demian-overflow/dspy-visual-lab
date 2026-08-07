from PIL import Image

from ..registry import register


@register(
    "extract_palette",
    "Extract dominant colors"
)
async def extract_palette(image, n=5):
    with Image.open(image) as img:
        rgb = img.convert("RGB")
        quantized = rgb.quantize(colors=n, method=Image.MEDIANCUT)
        palette = quantized.getpalette()[: n * 3]
        counts = sorted(quantized.getcolors(), reverse=True)

    colors = [
        "#{:02x}{:02x}{:02x}".format(
            palette[idx * 3], palette[idx * 3 + 1], palette[idx * 3 + 2]
        )
        for _count, idx in counts
    ]

    return {"colors": colors}

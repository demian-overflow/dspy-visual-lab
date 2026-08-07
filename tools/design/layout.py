from ..registry import register


def _all_elements(scene):
    return list(scene.get("objects", [])) + list(scene.get("text", []))


@register(
    "analyze_layout",
    "Analyze composition and spacing"
)
async def analyze_layout(scene):
    elements = _all_elements(scene)

    if len(elements) < 2:
        return {"grid": "unknown", "alignment": "unknown", "spacing": {}}

    lefts = [e["bbox"]["x"] for e in elements]
    rights = [e["bbox"]["x"] + e["bbox"]["width"] for e in elements]
    centers = [e["bbox"]["x"] + e["bbox"]["width"] / 2 for e in elements]

    tolerance = 5.0

    def _spread(values):
        return max(values) - min(values)

    spreads = {
        "left": _spread(lefts),
        "right": _spread(rights),
        "center": _spread(centers),
    }
    alignment = min(spreads, key=spreads.get)
    if spreads[alignment] > tolerance:
        alignment = "mixed"

    grid = "aligned" if alignment != "mixed" else "unaligned"

    sorted_by_y = sorted(elements, key=lambda e: e["bbox"]["y"])
    gaps = [
        sorted_by_y[i + 1]["bbox"]["y"]
        - (sorted_by_y[i]["bbox"]["y"] + sorted_by_y[i]["bbox"]["height"])
        for i in range(len(sorted_by_y) - 1)
    ]

    spacing = {"min_gap": min(gaps), "max_gap": max(gaps)} if gaps else {}

    return {"grid": grid, "alignment": alignment, "spacing": spacing}

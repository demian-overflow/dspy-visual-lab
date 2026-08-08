from difflib import SequenceMatcher


def text_similarity(
    a: str,
    b: str
):

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()



def color_similarity(
    original,
    generated
):

    if not original:
        return 0

    matches = 0

    for color in original:

        if color in generated:
            matches += 1

    return matches / len(original)



def _coord(box, key):
    """Read a bbox field from either an object or a mapping."""
    if isinstance(box, dict):
        return box[key]
    return getattr(box, key)


def bbox_similarity(
    a,
    b,
    canvas_width=None,
    canvas_height=None
):
    """Similarity of two bboxes, normalized by the canvas they live on.

    Deltas are expressed as fractions of the canvas, so the same *relative*
    error scores the same on a 994x1536 image as on a 2999x3838 one. A fixed
    pixel constant made this wildly scale-dependent.
    """

    width = canvas_width or 1
    height = canvas_height or 1

    dx = abs(_coord(a, "x") - _coord(b, "x")) / width
    dy = abs(_coord(a, "y") - _coord(b, "y")) / height

    dw = abs(_coord(a, "width") - _coord(b, "width")) / width
    dh = abs(_coord(a, "height") - _coord(b, "height")) / height

    return 1 - (
        dx+dy+dw+dh
    ) / 4


def _bbox_center(bbox):
    return (bbox["x"] + bbox["width"] / 2, bbox["y"] + bbox["height"] / 2)


def _center_distance(a, b):
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def match_by_bbox(gold_items, pred_items):
    if not pred_items:
        return [(g, None) for g in gold_items]

    pairs = []
    for g in gold_items:
        nearest = min(pred_items, key=lambda p: _center_distance(g["bbox"], p["bbox"]))
        pairs.append((g, nearest))
    return pairs

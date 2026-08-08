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



def bbox_similarity(
    a,
    b
):

    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y)

    dw = abs(
        a.width-b.width
    )

    dh = abs(
        a.height-b.height
    )

    return 1 - (
        dx+dy+dw+dh
    ) / 4000


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

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

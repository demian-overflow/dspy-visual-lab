import dspy


class ExtractScene(dspy.Signature):

    """
    Analyze an image and produce
    structured scene JSON.
    """

    image: dspy.Image = dspy.InputField(
        desc="input image"
    )


    scene = dspy.OutputField(
        desc="""
        JSON object with keys:
        width (image pixel width),
        height (image pixel height),
        background (hex),
        objects,
        text,
        layout,
        colors.
        Bounding boxes are {x, y, width, height} in pixels.
        """
    )

import dspy


class ExtractScene(dspy.Signature):

    """
    Analyze an image and produce
    structured scene JSON.
    """

    image = dspy.InputField(
        desc="input image"
    )


    scene = dspy.OutputField(
        desc="""
        JSON:
        objects,
        text,
        layout,
        colors,
        typography
        """
    )

import dspy


class RenderCreative(dspy.Signature):

    plan = dspy.InputField()

    artifact = dspy.OutputField(
        desc="""
        HTML/SVG/image generation instructions
        """
    )

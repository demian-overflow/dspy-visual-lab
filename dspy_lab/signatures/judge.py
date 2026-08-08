import dspy


class JudgeCreative(dspy.Signature):

    original = dspy.InputField()

    generated = dspy.InputField()


    score = dspy.OutputField(
        desc="""
        JSON:
        {
          layout,
          typography,
          colors,
          objects,
          overall
        }
        """
    )

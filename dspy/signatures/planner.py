import dspy


class CreatePlan(dspy.Signature):

    scene = dspy.InputField()

    plan = dspy.OutputField(
        desc="""
        Ordered tool execution plan.
        Each step contains:
        tool name,
        arguments,
        expected output
        """
    )

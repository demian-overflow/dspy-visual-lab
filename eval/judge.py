import dspy


class VisionJudge(dspy.Module):


    def __init__(
        self,
        signature
    ):

        self.judge = dspy.Predict(
            signature
        )


    def forward(
        self,
        original,
        generated
    ):

        return self.judge(
            original=original,
            generated=generated
        )

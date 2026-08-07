import dspy

from ..signatures.judge import JudgeCreative


class CreativeCritic(dspy.Module):

    def __init__(self):

        self.judge = dspy.Predict(
            JudgeCreative
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

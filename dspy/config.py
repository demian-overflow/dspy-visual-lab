import dspy


def configure(lm):

    dspy.settings.configure(
        lm=lm
    )

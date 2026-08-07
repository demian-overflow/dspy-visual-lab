import dspy


def optimize(
    program,
    examples
):

    optimizer = dspy.MIPROv2(
        metric=None,
        auto="light"
    )


    return optimizer.compile(
        program,
        trainset=examples
    )

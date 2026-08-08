import dspy

from eval.scorer import parser_metric


def optimize(program, examples, metric=None):
    optimizer = dspy.MIPROv2(
        metric=metric or parser_metric,
        auto="light"
    )

    return optimizer.compile(
        program,
        trainset=examples
    )

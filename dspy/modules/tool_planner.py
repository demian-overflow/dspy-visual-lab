import dspy

from ..signatures.planner import CreatePlan


class ToolPlanner(dspy.Module):

    def __init__(self):

        self.plan = dspy.ChainOfThought(
            CreatePlan
        )


    def forward(
        self,
        scene
    ):

        return self.plan(
            scene=scene
        )

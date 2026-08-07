import dspy

from .scene_parser import SceneParser
from .tool_planner import ToolPlanner


class CreativePipeline(dspy.Module):

    def __init__(self):

        self.parser = SceneParser()

        self.planner = ToolPlanner()


    def forward(
        self,
        image
    ):

        scene = self.parser(
            image=image
        )


        plan = self.planner(
            scene=scene.scene
        )


        return dspy.Prediction(
            scene=scene,
            plan=plan
        )

import dspy
from ..signatures.scene import ExtractScene


class SceneParser(dspy.Module):

    def __init__(self):

        self.parser = dspy.ChainOfThought(
            ExtractScene
        )


    def forward(
        self,
        image
    ):

        return self.parser(
            image=image
        )

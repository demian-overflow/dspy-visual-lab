from pathlib import Path

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

        # ExtractScene.image is a dspy.Image so the bytes actually reach the
        # model; callers that still hand over a path (e.g. CreativeAgent) are
        # coerced here rather than silently sending the path as text.
        if isinstance(image, (str, Path)):
            image = dspy.Image.from_path(str(image))

        return self.parser(
            image=image
        )

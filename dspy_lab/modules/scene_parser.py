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

        result = self.parser(
            image=image
        )

        # ExtractScene.scene is a typed Scene pydantic field (for reliable
        # structured output); callers (SceneScorer, CreativeAgent, the
        # entry-point script) all expect a plain dict/JSON string, so
        # convert once here rather than at every call site.
        result.scene = result.scene.model_dump()

        return result

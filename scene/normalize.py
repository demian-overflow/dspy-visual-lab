from .schema import Scene


class SceneNormalizer:


    def normalize(
        self,
        scene: Scene
    ):

        for color in scene.colors:

            color.hex = (
                color.hex.lower()
            )

        return scene

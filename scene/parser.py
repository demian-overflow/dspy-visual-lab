from .schema import Scene


class SceneParser:


    def from_json(
        self,
        data: dict
    ) -> Scene:

        return Scene(
            **data
        )


    def from_tools(
        self,
        results:list
    ):

        scene={}

        for item in results:

            scene.update(
                item["result"]
            )

        return Scene(
            **scene
        )

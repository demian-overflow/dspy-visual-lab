from .schema import Scene


class SceneMerger:


    def merge(
        self,
        scenes:list[Scene]
    ):

        result = scenes[0]


        for scene in scenes[1:]:

            result.objects.extend(
                scene.objects
            )

            result.text.extend(
                scene.text
            )

            result.colors.extend(
                scene.colors
            )


        return result

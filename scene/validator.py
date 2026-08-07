from .schema import Scene


class SceneValidator:


    def validate(
        self,
        scene: Scene
    ):

        errors=[]


        if scene.width <= 0:
            errors.append(
                "invalid width"
            )


        if scene.height <=0:
            errors.append(
                "invalid height"
            )


        for obj in scene.objects:

            if obj.bbox.x < 0:
                errors.append(
                    f"{obj.id}: bad x"
                )


        return {
            "valid":
                len(errors)==0,

            "errors":
                errors
        }

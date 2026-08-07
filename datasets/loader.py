from .schema import CreativeSample


class DatasetLoader:


    def load(
        self,
        manifest
    ):

        return [
            CreativeSample(
                **item
            )
            for item in manifest
        ]

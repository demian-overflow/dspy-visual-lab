class SceneScorer:


    WEIGHTS = {

        "text":0.25,

        "objects":0.25,

        "layout":0.20,

        "colors":0.15,

        "aesthetic":0.15
    }



    def score(
        self,
        result
    ):

        total = 0


        for key, weight in self.WEIGHTS.items():

            total += (
                result[key]
                *
                weight
            )


        return total

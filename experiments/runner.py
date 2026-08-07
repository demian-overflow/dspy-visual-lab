from datetime import datetime


class ExperimentRunner:


    def __init__(
        self,
        pipeline,
        evaluator
    ):

        self.pipeline = pipeline

        self.evaluator = evaluator



    async def run(
        self,
        experiment,
        dataset
    ):

        results=[]


        for sample in dataset:

            output = await self.pipeline.recreate(
                sample.image_path
            )


            score = self.evaluator.score(
                output
            )


            results.append(
                {
                    "sample":
                        sample.id,

                    "score":
                        score
                }
            )


        return {

            "experiment":
                experiment.name,

            "timestamp":
                datetime.utcnow()
                .isoformat(),

            "results":
                results
        }

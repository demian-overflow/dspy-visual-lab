import json
from pathlib import Path


class ExperimentTracker:


    def save(
        self,
        experiment_name,
        result
    ):

        path = Path(
            "experiments/results"
        )

        path.mkdir(
            parents=True,
            exist_ok=True
        )


        file = (
            path /
            f"{experiment_name}.json"
        )


        file.write_text(
            json.dumps(
                result,
                indent=2
            )
        )

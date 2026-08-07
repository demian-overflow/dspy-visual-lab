import json
from datetime import datetime


class ExperimentReport:


    def save(
        self,
        name,
        metrics
    ):

        data = {

            "experiment":name,

            "time":
                datetime.utcnow()
                .isoformat(),

            "metrics":
                metrics
        }


        with open(
            "report.json",
            "w"
        ) as f:

            json.dump(
                data,
                f,
                indent=2
            )

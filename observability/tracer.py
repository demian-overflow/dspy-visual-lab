import time


class Trace:


    def __init__(
        self,
        run_id
    ):

        self.run_id = run_id

        self.steps = []



    def start(
        self,
        name,
        data=None
    ):

        self.steps.append(
            {
                "name": name,
                "start": time.time(),
                "input": data
            }
        )



    def end(
        self,
        output=None
    ):

        step = self.steps[-1]

        step["end"] = time.time()

        step["duration"] = (
            step["end"]
            -
            step["start"]
        )

        step["output"] = output



    def export(self):

        return {
            "run_id": self.run_id,
            "steps": self.steps
        }

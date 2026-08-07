import uuid

from .paths import run_path


class RunStorage:


    def create_run(self):

        run_id = (
            str(uuid.uuid4())
        )

        path = run_path(
            run_id
        )

        return {
            "id": run_id,
            "path": path
        }

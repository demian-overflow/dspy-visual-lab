import json

from pathlib import Path



def load_manifest(
    path: str
):

    data = json.loads(
        Path(path)
        .read_text()
    )

    return data

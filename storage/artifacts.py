import json
from pathlib import Path


class ArtifactStorage:


    def save_json(
        self,
        path: Path,
        name: str,
        data: dict
    ):

        file = path / name

        file.write_text(
            json.dumps(
                data,
                indent=2
            )
        )

        return file



    def save_text(
        self,
        path: Path,
        name: str,
        content: str
    ):

        file = path / name

        file.write_text(
            content
        )

        return file



    def save_bytes(
        self,
        path: Path,
        name: str,
        data: bytes
    ):

        file = path / name

        file.write_bytes(
            data
        )

        return file

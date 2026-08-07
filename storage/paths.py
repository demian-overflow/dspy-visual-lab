from pathlib import Path


ROOT = Path("runs")


def run_path(run_id: str) -> Path:

    path = ROOT / run_id

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    return path

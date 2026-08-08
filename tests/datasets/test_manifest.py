import json
from pathlib import Path

from PIL import Image

from datasets.loader import DatasetLoader
from datasets.manifest import load_manifest

DATASETS_ROOT = Path(__file__).resolve().parents[2] / "datasets"


def test_manifest_entries_resolve_to_real_files():
    manifest = load_manifest(str(DATASETS_ROOT / "manifests" / "train.json"))
    samples = DatasetLoader().load(manifest)

    assert len(samples) >= 2
    for sample in samples:
        image_path = DATASETS_ROOT / sample.image_path
        scene_path = DATASETS_ROOT / sample.scene_path
        assert image_path.exists(), f"missing {image_path}"
        assert scene_path.exists(), f"missing {scene_path}"

        with Image.open(image_path) as img:
            assert img.size[0] > 0 and img.size[1] > 0

        scene_data = json.loads(scene_path.read_text())
        assert scene_data["width"] > 0
        assert scene_data["height"] > 0
        assert len(scene_data["text"]) >= 1

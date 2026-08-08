import json
from pathlib import Path
from unittest.mock import patch

from datasets.schema import CreativeSample
from scripts.run_experiment import run


def test_run_scores_each_sample_against_ground_truth(tmp_path):
    scene_json = {
        "width": 10, "height": 10, "background": "#fff",
        "objects": [], "text": [], "colors": [{"hex": "#ffffff"}],
    }
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(scene_json))

    image_path = tmp_path / "image.png"
    from PIL import Image
    Image.new("RGB", (10, 10)).save(image_path)

    sample = CreativeSample(
        id="s1",
        image_path="image.png",
        category="poster",
        scene_path="scene.json",
    )

    with patch("scripts.run_experiment.SceneParser") as mock_parser_cls:
        mock_parser_cls.return_value.return_value.scene = json.dumps(scene_json)

        result = run([sample], dataset_root=tmp_path, optimize=False)

    assert result["results"][0]["sample"] == "s1"
    assert result["results"][0]["score"] > 0.9  # identical scene => near-perfect score
